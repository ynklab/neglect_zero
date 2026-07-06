import os
import logging
import argparse
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.integrations import HfDeepSpeedConfig
from transformers.modeling_utils import PreTrainedModel
import myutil
import csv

instruction = """In the following, we will ask for your judgments about certain kinds of sentences in English.
The sentences refers to pairs of pictures.
The picture contains two types of geometrical shapes, one in the upper half and one in the lower half of the picture.
One of these shapes in the picture were homogeneous with respect to their color, and the other set had mixed colors, containing one element with a different color.
Only one of the pictures would match with the sentence in each trial, and your task is to choose that one.
The covered picture, what we call "better-picture", is sometimes contained in the pairs.

You will see many pairs, each of which will be accompanied by an sentence about the contents of the pictures.
Your task is to decide which picture in a pair match this sentence.
The better-picture should only be chosen if the open picture did not match the sentence.
You will answer 'First' if you consider the picture 1 match the sentence; otherwise you will answer 'Second'.
Do not include any words or sentences other than "First" or "Second" in your answer.

You will start with a short training to get you familiar with the response procedure.
During this training, you will see examples of correct responses.
### Training 1
{fewshot_1}

### Training 2
{fewshot_2}

### Training 3
{fewshot_3}

### Training 4
{fewshot_4}

### Training 5
{fewshot_5}

### Training 6
{fewshot_6}

### Training 7
{fewshot_7}

### Training 8
{fewshot_8}

### Your turn
As in the training, you will decide which pictures are appropriate to the sentence you see.

"""

filler_path = "materials/FILLER_stimuli_table(neckar).csv"
esq_path = "materials/ESQ_stimuli_table(lemberg).csv"
dis_path = "materials/DIST_stimuli_table(hochberg).csv"
upp_path = "materials/UPP_stimuli_table(oberhohenberg).csv"
bas_path = "materials/BAS_stimuli_table(rainen).csv"

master_seed = 12345
default_seed = myutil.make_default_seed(master_seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Llama-4-Scout Inference Test")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                        help="Path or name of the model")
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument("--seeds", type=int, nargs='+', default=default_seed,
                        help="List of random seeds (one per subject)")
    parser.add_argument("--batch_id", type=int, default=None, choices=range(1, 17),
                        help="Batch ID (1-16) to run 20 seeds at a time. If not specified, runs all seeds.")
    deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def load_model_zero(args, logger):
    # Initialization of DeepSpeed Config
    if hasattr(args, 'deepspeed_config') and args.deepspeed_config:
        hfdsc = HfDeepSpeedConfig(args.deepspeed_config)

    model_config_args = {
        'pretrained_model_name_or_path': args.model_name_or_path,
        'dtype': torch.float16,
        'trust_remote_code': True,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(**model_config_args)
    logger.info(f"Successfully loaded the model.")
    logger.info(f'Using memory: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB')

    # DeepSpeed Initialize
    model, *_ = deepspeed.initialize(
        model=model,
        args=args,
    )
   
    model.eval()
   
    return model, tokenizer


def run_inference(prompt, model, tokenizer, logger, messages, rank=0):
    if rank == 0:
        logger.info(f"executed prompt: {prompt[:50]}...")

    messages.append({"role": "user", "content": prompt})
   
    text_input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(text_input, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
   
    input_length = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            synced_gpus=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    if rank == 0:
        logger.info(f"Generated text: {generated_text}")

    messages.append({"role": "assistant", "content": generated_text})

    logger.info(f"Using memory of RANK {rank}: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
   
    return generated_text


def main():
    args = parse_args()

    # For distributed training, get the world size and rank from environment variables
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
   
    # Initialization of DeepSpeed's distributed initialization
    deepspeed.init_distributed(dist_backend="nccl", world_size=world_size, rank=world_rank)

    logging.basicConfig(
        format=f"[{world_rank}/{world_size}] %(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    if world_rank != 0:
        logger.setLevel(logging.WARNING)

    logger.info(f"Process {world_rank}/{world_size} has started.")
   
    if args.batch_id is not None:
        batch_size = 20
        start_idx = (args.batch_id - 1) * batch_size
        end_idx = start_idx + batch_size
       
        if start_idx >= len(args.seeds):
            if world_rank == 0:
                logger.error("There is no seed corresponding to the specified batch ID.")
            return

        target_seeds = args.seeds[start_idx : end_idx]
       
        if world_rank == 0:
            logger.info(f"Batch execution mode: Batch ID {args.batch_id}")
            logger.info(f"Executing seeds from index {start_idx} to {end_idx-1}: {len(target_seeds)} seeds.")
    else:
        target_seeds = args.seeds
        start_idx = 0
        if world_rank == 0:
            logger.info(f"Normal execution mode: Executing all {len(target_seeds)} seeds.")
   
    def apply_deepspeed_initialization_patch():
        original_init_weights = PreTrainedModel._init_weights

        def safe_init_weights(self, module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                if module.weight.shape[0] == 0:
                    return
       
            try:
                original_init_weights(self, module)
            except IndexError:
                pass

        PreTrainedModel._init_weights = safe_init_weights
        print("[INFO] Applied DeepSpeed initialization patch for embeddings.")

    apply_deepspeed_initialization_patch()
   
    model, tokenizer = load_model_zero(args, logger)
   
    for local_idx, seed in enumerate(target_seeds):
        set_seed(seed)
       
        subject_idx = local_idx + start_idx
       
        # ≡0: BAS, ≡1: ESQ, ≡2: DIST, ≡3: UPP (mod 4)
        exp_type_idx = subject_idx % 4
        exp_number = subject_idx // 4 + 1
       
        if exp_type_idx == 0:
            subject_id = f"BAS_Subject{exp_number}"
            output_path = f"subject_data/llama4scout/experiment_list_{subject_id}.csv"
            target_csv = bas_path
            is_bas = True
        elif exp_type_idx == 1:
            subject_id = f"ESQ_Subject{exp_number}"
            output_path = f"subject_data/llama4scout/experiment_list_{subject_id}.csv"
            target_csv = esq_path
            is_bas = False
        elif exp_type_idx == 2:
            subject_id = f"DIST_Subject{exp_number}"
            output_path = f"subject_data/llama4scout/experiment_list_{subject_id}.csv"
            target_csv = dis_path
            is_bas = False
        else:
            subject_id = f"UPP_Subject{exp_number}"
            output_path = f"subject_data/llama4scout/experiment_list_{subject_id}.csv"
            target_csv = upp_path
            is_bas = False

        if world_rank == 0:
            try:
                myutil.make_experiment_csv(filler_path, target_csv, output_path, seed, bas_flag=is_bas)
                logger.info(f"Rank 0: Created experiment CSV at {output_path}")
            except Exception as e:
                logger.error(f"Failed to create experiment CSV: {e}")
        dist.barrier()
       
        try:
            item_list = myutil.converter(output_path, seed)
        except Exception as e:
            if world_rank == 0: logger.error(f"Converter failed: {e}")
            item_list = []
       
        prompts = myutil.make_prompts(item_list, instruction)
       
        dist.barrier()
   
        if world_rank == 0:
            logger.info(f"========== Start Subject: {subject_id} (Seed: {seed}) ==========")

        messages = []
        results = []
       
        for i, p in enumerate(prompts):
           
            if world_rank == 0:
                logger.info(f"--- {subject_id} / Round {i+1} ---")
               
            generated_text = run_inference(p[0], model, tokenizer, logger, messages, rank=world_rank)
           
            dist.barrier()

            if world_rank == 0:
                is_correct = myutil.check_correctness(generated_text, p[1], p[2])
               
                result_row = {
                    "Item_ID": p[3],
                    "Condition": p[1],
                    "Correct_Answer": p[2],
                    "LLM_Response": generated_text,
                    "Is_Correct": is_correct
                }
                results.append(result_row)
               
                logger.info(f"Trial {i}: {p[1]} -> Correct? {is_correct}")
               
        if world_rank == 0:
            try:
                fieldnames = ["Item_ID", "Condition", "Correct_Answer", "LLM_Response", "Is_Correct"]
                # You have to ensure that the 'results' directory exists before writing the CSV file
                output_filename = f"results/llama4scout/result_{subject_id}.csv"
                with open(output_filename, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(results)
                   
                logger.info(f"Saved results to {output_filename}")
            except Exception as e:
                logger.error(f"Failed to save CSV: {e}")

        dist.barrier()

if __name__ == "__main__":
    main()