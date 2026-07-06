import logging
import argparse
import csv
import os
import random
from openai import OpenAI
import myutil

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

client = OpenAI()

def parse_args():
    parser = argparse.ArgumentParser(description="GPT-5 nano Inference Test")
    parser.add_argument("--model_name", type=str, default="gpt-5-nano-2025-08-07",
                        help="Name of the OpenAI model to use")
    parser.add_argument("--seeds", type=int, nargs='+', default=default_seed,
                        help="List of random seeds (one per subject)")
    parser.add_argument("--batch_id", type=int, default=None, choices=range(1, 17),
                        help="Batch ID (1-16) to run 20 seeds at a time. If not specified, runs all seeds.")
    return parser.parse_args()


def run_inference(prompt, model_name, logger, messages, seed_val):
    logger.info(f"Executing prompt: {prompt[:50]}...")

    messages.append({"role": "user", "content": prompt})
   
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1,
            seed=seed_val
        )
        generated_text = response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API Error: {e}")
        generated_text = "Error"

    logger.info(f"Generated text: {generated_text}")
    messages.append({"role": "assistant", "content": generated_text})
   
    return generated_text


def main():
    args = parse_args()

    logging.basicConfig(
        format=f"%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    if args.batch_id is not None:
        batch_size = 20
        start_idx = (args.batch_id - 1) * batch_size
        end_idx = start_idx + batch_size
        if start_idx >= len(args.seeds):
            logger.error("There is no specified batch ID corresponding to the seeds.")
            return
        target_seeds = args.seeds[start_idx : end_idx]
        logger.info(f"Batch execution mode: Batch ID {args.batch_id}")
        logger.info(f"Executing seeds from index {start_idx} to {end_idx-1}: {len(target_seeds)} seeds.")
    else:
        target_seeds = args.seeds
        start_idx = 0
        logger.info(f"Normal execution mode: Running all {len(target_seeds)} seeds.")
   
    logger.info(f"Using model: {args.model_name}")

    for local_idx, seed in enumerate(target_seeds):
        random.seed(seed)
        
        subject_idx = local_idx + start_idx
       
        # ≡0: BAS, ≡1: ESQ, ≡2: DIST, ≡3: UPP (mod 4)
        exp_type_idx = subject_idx % 4
        exp_number = subject_idx // 4 + 1
       
        out_dir = "subject_data/gpt5nano"
        os.makedirs(out_dir, exist_ok=True)
        
        if exp_type_idx == 0:
            subject_id = f"BAS_Subject{exp_number}"
            output_path = f"{out_dir}/experiment_list_{subject_id}.csv"
            target_csv = bas_path
            is_bas = True
        elif exp_type_idx == 1:
            subject_id = f"ESQ_Subject{exp_number}"
            output_path = f"{out_dir}/experiment_list_{subject_id}.csv"
            target_csv = esq_path
            is_bas = False
        elif exp_type_idx == 2:
            subject_id = f"DIST_Subject{exp_number}"
            output_path = f"{out_dir}/experiment_list_{subject_id}.csv"
            target_csv = dis_path
            is_bas = False
        else:
            subject_id = f"UPP_Subject{exp_number}"
            output_path = f"{out_dir}/experiment_list_{subject_id}.csv"
            target_csv = upp_path
            is_bas = False

        try:
            myutil.make_experiment_csv(filler_path, target_csv, output_path, seed, bas_flag=is_bas)
            logger.info(f"Created experiment CSV at {output_path}")
        except Exception as e:
            logger.error(f"Failed to create experiment CSV: {e}")
            continue
       
        item_list = myutil.converter(output_path, seed)
        prompts = myutil.make_prompts(item_list, instruction)
        logger.info(f"========== Start Subject: {subject_id} (Seed: {seed}) ==========")

        messages = []
        results = []
       
        for i, p in enumerate(prompts):
            logger.info(f"--- {subject_id} / Round {i+1} ---")
            
            generated_text = run_inference(p[0], args.model_name, logger, messages, seed)
            
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
               
        try:
            res_dir = "results/gpt5nano"
            os.makedirs(res_dir, exist_ok=True)
            
            fieldnames = ["Item_ID", "Condition", "Correct_Answer", "LLM_Response", "Is_Correct"]
            output_filename = f"{res_dir}/result_{subject_id}.csv"
            with open(output_filename, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            logger.info(f"Saved results to {output_filename}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

if __name__ == "__main__":
    main()