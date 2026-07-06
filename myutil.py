import pandas as pd
import random
import csv

def make_default_seed(master_seed, num_subjects=320):
    rng = random.Random(master_seed)

    seeds = rng.sample(range(1000000), num_subjects)
    
    return seeds

def make_experiment_csv(filler_csv_path, priming_csv_path, output_csv_path, subject_seed, bas_flag=False):
    
    random.seed(subject_seed)
    print(f"Subject seed value {subject_seed} is used to create the experiment item list for the follow-up experiment.\n")
    
    try:
        df_filler = pd.read_csv(filler_csv_path)
        df_priming = pd.read_csv(priming_csv_path)
    except FileNotFoundError as e:
        print(f"Error: file not found - {e}")
        return None

    ordered_fillers = df_filler.copy().reset_index(drop=True)

    experiment_sequence = []
    
    id = 0
    fewshots_subset = ordered_fillers.iloc[-8:].copy()
    
    fewshots_subset['trial_type'] = 'fewshot'
    fewshots_subset['set_id'] = id
    for _, row in fewshots_subset.iterrows():
        experiment_sequence.append(row)

    for n in range(48):
        filler_start_idx = n * 3
        
        fillers_subset = ordered_fillers.iloc[filler_start_idx : filler_start_idx + 3].copy()
        fillers_subset['trial_type'] = 'filler'
        fillers_subset['set_id'] = n + 1
        for _, row in fillers_subset.iterrows():
            experiment_sequence.append(row)

        base_start_1based = 8 * n + 2
        candidate_starts_1based = [
            base_start_1based,
            base_start_1based + 2,
            base_start_1based + 4,
            base_start_1based + 6
        ]
        
        selected_start_1based = random.choice(candidate_starts_1based)
        
        prime_idx = selected_start_1based - 2
        target_idx = selected_start_1based - 1
        
        prime_row = df_priming.iloc[prime_idx].copy()
        target_row = df_priming.iloc[target_idx].copy()

        if bas_flag:
            prime_row["trial_type"] = "bas_prime"
        elif selected_start_1based in [base_start_1based, base_start_1based + 4]:
            prime_row['trial_type'] = 'critical_prime'
        else:
            prime_row['trial_type'] = 'control_prime'
            
        prime_row['set_id'] = n + 1
        target_row['trial_type'] = 'target'
        target_row['set_id'] = n + 1

        experiment_sequence.append(prime_row)
        experiment_sequence.append(target_row)

    df_result = pd.DataFrame(experiment_sequence)
    df_result = df_result.convert_dtypes()
    df_result.to_csv(output_csv_path, index=False)
    
    print(f"finished: {len(df_result)} rows created and saved to '{output_csv_path}'.")
    return df_result


def get_shuffled_options(rng):
    options = ["First", "Second"]
    return rng.sample(options, len(options))


def converter(filename, subject_seed):
    rng = random.Random(subject_seed)
    
    template1_1 = "Picture 1 contains {n_colour1_top_1} {colour1_top_1} {shape_top_1}{pl1_top} and {n_colour2_top_1} {colour2_top_1} {shape_top_1}{pl2_top} in the upper half, and {n_colour1_bottom_1} {colour1_bottom_1} {shape_bottom_1}{pl1_bottom} and {n_colour2_bottom_1} {colour2_bottom_1} {shape_bottom_1}{pl2_bottom} in the lower half.\n"
    template1_2 = "Picture 1 contains {n_colour_top_1} {colour_top_1} {shape_top_1}{pl_top} in the upper half, and {n_colour1_bottom_1} {colour1_bottom_1} {shape_bottom_1}{pl1_bottom} and {n_colour2_bottom_1} {colour2_bottom_1} {shape_bottom_1}{pl2_bottom} in the lower half.\n"
    template1_3 = "Picture 1 contains {n_colour1_top_1} {colour1_top_1} {shape_top_1}{pl1_top} and {n_colour2_top_1} {colour2_top_1} {shape_top_1}{pl2_top} in the upper half, and {n_colour_bottom_1} {colour_bottom_1} {shape_bottom_1}{pl_bottom} in the lower half.\n"
    template1_4 = "Picture 1 is the better-picture.\n"
    template2_1 = "Picture 2 contains {n_colour1_top_2} {colour1_top_2} {shape_top_2}{pl1_top} and {n_colour2_top_2} {colour2_top_2} {shape_top_2}{pl2_top} in the upper half, and {n_colour1_bottom_2} {colour1_bottom_2} {shape_bottom_2}{pl1_bottom} and {n_colour2_bottom_2} {colour2_bottom_2} {shape_bottom_2}{pl2_bottom} in the lower half.\n"
    template2_2 = "Picture 2 contains {n_colour_top_2} {colour_top_2} {shape_top_2}{pl_top} in the upper half, and {n_colour1_bottom_2} {colour1_bottom_2} {shape_bottom_2}{pl1_bottom} and {n_colour2_bottom_2} {colour2_bottom_2} {shape_bottom_2}{pl2_bottom} in the lower half.\n"
    template2_3 = "Picture 2 contains {n_colour1_top_2} {colour1_top_2} {shape_top_2}{pl1_top} and {n_colour2_top_2} {colour2_top_2} {shape_top_2}{pl2_top} in the upper half, and {n_colour_bottom_2} {colour_bottom_2} {shape_bottom_2}{pl_bottom} in the lower half.\n"
    template2_4 = "Picture 2 is the better-picture.\n"
    template3 = "The sentence is: {sentence}\n"
    template4 = "Which picture does match with the sentence in this situation? Here are your answer options:\n{option1}\n{option2}\n"
    message1 = ""
    message2 = ""
    message3 = ""
    message4 = ""
    
    experiment_list = []
    
    try:
        with open(filename, mode="r", encoding="utf-8") as f:
            number = 2
            reader = csv.DictReader(f)
            print("example")
            for row in reader:
                # Picture1
                if row["shape_top_1"] == "better":
                    message1 = template1_4
                elif row["n_colour1_top_1"] == "0":
                    if row["shape_top_1"] == "cross":
                        pl2_top = "es"
                    else:
                        pl2_top = "s"
                    if row["n_colour1_bottom_1"] == "1":
                        pl1_bottom = ""
                        if row["shape_bottom_1"] == "cross":
                            pl2_bottom = "es"
                        else:
                            pl2_bottom = "s"
                    elif row["n_colour2_bottom_1"] == "1":
                        if row["shape_bottom_1"] == "cross":
                            pl1_bottom = "es"
                        else:
                            pl1_bottom = "s"
                        pl2_bottom = ""
                    else:
                        if row["shape_bottom_1"] == "cross":
                            pl1_bottom = "es"
                            pl2_bottom = "es"
                        else:
                            pl1_bottom = "s"
                            pl2_bottom = "s"
                    message1 = template1_2.format(
                        n_colour_top_1=row["n_colour2_top_1"],
                        colour_top_1=row["colour2_top_1"],
                        pl_top=pl2_top,
                        shape_top_1=row["shape_top_1"],
                        n_colour1_bottom_1=row["n_colour1_bottom_1"],
                        colour1_bottom_1=row["colour1_bottom_1"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_1=row["n_colour2_bottom_1"],
                        colour2_bottom_1=row["colour2_bottom_1"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_1=row["shape_bottom_1"]
                    )
                elif row["n_colour2_top_1"] == "0":
                    if row["shape_top_1"] == "cross":
                        pl1_top = "es"
                    else:
                        pl1_top = "s"
                    if row["n_colour1_bottom_1"] == "1":
                        pl1_bottom = ""
                        if row["shape_bottom_1"] == "cross":
                            pl2_bottom = "es"
                        else:
                            pl2_bottom = "s"
                    elif row["n_colour2_bottom_1"] == "1":
                        if row["shape_bottom_1"] == "cross":
                            pl1_bottom = "es"
                        else:
                            pl1_bottom = "s"
                        pl2_bottom = ""
                    else:
                        if row["shape_bottom_1"] == "cross":
                            pl1_bottom = "es"
                            pl2_bottom = "es"
                        else:
                            pl1_bottom = "s"
                            pl2_bottom = "s"
                    message1 = template1_2.format(
                        n_colour_top_1=row["n_colour1_top_1"],
                        colour_top_1=row["colour1_top_1"],
                        pl_top=pl1_top,
                        shape_top_1=row["shape_top_1"],
                        n_colour1_bottom_1=row["n_colour1_bottom_1"],
                        colour1_bottom_1=row["colour1_bottom_1"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_1=row["n_colour2_bottom_1"],
                        colour2_bottom_1=row["colour2_bottom_1"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_1=row["shape_bottom_1"]
                    )
                elif row["n_colour1_bottom_1"] == "0":
                    if row["shape_bottom_1"] == "cross":
                        pl2_bottom = "es"
                    else:
                        pl2_bottom = "s"
                    if row["n_colour1_top_1"] == "1":
                        pl1_top = ""
                        if row["shape_top_1"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                    elif row["n_colour2_top_1"] == "1":
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                    else:
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                    message1 = template1_3.format(
                        n_colour1_top_1=row["n_colour1_top_1"],
                        colour1_top_1=row["colour1_top_1"],
                        pl1_top=pl1_top,
                        n_colour2_top_1=row["n_colour2_top_1"],
                        colour2_top_1=row["colour2_top_1"],
                        pl2_top=pl2_top,
                        shape_top_1=row["shape_top_1"],
                        n_colour_bottom_1=row["n_colour2_bottom_1"],
                        colour_bottom_1=row["colour2_bottom_1"],
                        pl_bottom=pl2_bottom,
                        shape_bottom_1=row["shape_bottom_1"]
                    )
                elif row["n_colour2_bottom_1"] == "0":
                    if row["shape_bottom_1"] == "cross":
                        pl1_bottom = "es"
                    else:
                        pl1_bottom = "s"
                    if row["n_colour1_top_1"] == "1":
                        pl1_top = ""
                        if row["shape_top_1"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                    elif row["n_colour2_top_1"] == "1":
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                    else:
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                    message1 = template1_3.format(
                        n_colour1_top_1=row["n_colour1_top_1"],
                        colour1_top_1=row["colour1_top_1"],
                        pl1_top=pl1_top,
                        n_colour2_top_1=row["n_colour2_top_1"],
                        colour2_top_1=row["colour2_top_1"],
                        pl2_top=pl2_top,
                        shape_top_1=row["shape_top_1"],
                        n_colour_bottom_1=row["n_colour2_bottom_1"],
                        colour_bottom_1=row["colour2_bottom_1"],
                        pl_bottom=pl1_bottom,
                        shape_bottom_1=row["shape_bottom_1"]
                    )
                else:
                    if row["n_colour1_top_1"] == "1":
                        pl1_top = ""
                        if row["shape_top_1"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                        if row["n_colour1_bottom_1"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_1"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_1"] == "1":
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    elif row["n_colour2_top_1"] == "1":
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                        if row["n_colour1_bottom_1"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_1"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_1"] == "1":
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:  
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    else:
                        if row["shape_top_1"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                        if row["n_colour1_bottom_1"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_1"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_1"] == "1":
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:
                            if row["shape_bottom_1"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    message1 = template1_1.format(
                        n_colour1_top_1=row["n_colour1_top_1"],
                        colour1_top_1=row["colour1_top_1"],
                        pl1_top=pl1_top,
                        n_colour2_top_1=row["n_colour2_top_1"],
                        colour2_top_1=row["colour2_top_1"],
                        pl2_top=pl2_top,
                        shape_top_1=row["shape_top_1"],
                        n_colour1_bottom_1=row["n_colour1_bottom_1"],
                        colour1_bottom_1=row["colour1_bottom_1"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_1=row["n_colour2_bottom_1"],
                        colour2_bottom_1=row["colour2_bottom_1"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_1=row["shape_bottom_1"]
                    )
                # Picture2
                if row["shape_top_2"] == "better":
                    message2 = template2_4
                elif row["n_colour1_top_2"] == "0":
                    if row["shape_top_2"] == "cross":
                        pl2_top = "es"
                    else:
                        pl2_top = "s"
                    if row["n_colour1_bottom_2"] == "1":
                        pl1_bottom = ""
                        if row["shape_bottom_2"] == "cross":
                            pl2_bottom = "es"
                        else:
                            pl2_bottom = "s"
                    elif row["n_colour2_bottom_2"] == "1":
                        if row["shape_bottom_2"] == "cross":
                            pl1_bottom = "es"
                        else:
                            pl1_bottom = "s"
                        pl2_bottom = ""
                    else:
                        if row["shape_bottom_2"] == "cross":
                            pl1_bottom = "es"
                            pl2_bottom = "es"
                        else:
                            pl1_bottom = "s"
                            pl2_bottom = "s"
                    message2 = template2_2.format(
                        n_colour_top_2=row["n_colour2_top_2"],
                        colour_top_2=row["colour2_top_2"],
                        pl_top=pl2_top,
                        shape_top_2=row["shape_top_2"],
                        n_colour1_bottom_2=row["n_colour1_bottom_2"],
                        colour1_bottom_2=row["colour1_bottom_2"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_2=row["n_colour2_bottom_2"],
                        colour2_bottom_2=row["colour2_bottom_2"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_2=row["shape_bottom_2"]
                    )
                elif row["n_colour2_top_2"] == "0":
                    if row["shape_top_2"] == "cross":
                        pl1_top = "es"
                    else:
                        pl1_top = "s"
                    if row["n_colour1_bottom_2"] == "1":
                        pl1_bottom = ""
                        if row["shape_bottom_2"] == "cross":
                            pl2_bottom = "es"
                        else:
                            pl2_bottom = "s"
                    elif row["n_colour2_bottom_2"] == "1":
                        if row["shape_bottom_2"] == "cross":
                            pl1_bottom = "es"
                        else:
                            pl1_bottom = "s"
                        pl2_bottom = ""
                    else:
                        if row["shape_bottom_2"] == "cross":
                            pl1_bottom = "es"
                            pl2_bottom = "es"
                        else:
                            pl1_bottom = "s"
                            pl2_bottom = "s"
                    message2 = template2_2.format(
                        n_colour_top_2=row["n_colour1_top_2"],
                        colour_top_2=row["colour1_top_2"],
                        pl_top=pl1_top,
                        shape_top_2=row["shape_top_2"],
                        n_colour1_bottom_2=row["n_colour1_bottom_2"],
                        colour1_bottom_2=row["colour1_bottom_2"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_2=row["n_colour2_bottom_2"],
                        colour2_bottom_2=row["colour2_bottom_2"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_2=row["shape_bottom_2"]
                    )
                elif row["n_colour1_bottom_2"] == "0":
                    if row["shape_bottom_2"] == "cross":
                        pl2_bottom = "es"
                    else:
                        pl2_bottom = "s"
                    if row["n_colour1_top_2"] == "1":
                        pl1_top = ""
                        if row["shape_top_2"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                    elif row["n_colour2_top_2"] == "1":
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                    else:
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                    message2 = template2_3.format(
                        n_colour1_top_2=row["n_colour1_top_2"],
                        colour1_top_2=row["colour1_top_2"],
                        pl1_top=pl1_top,
                        n_colour2_top_2=row["n_colour2_top_2"],
                        colour2_top_2=row["colour2_top_2"],
                        pl2_top=pl2_top,
                        shape_top_2=row["shape_top_2"],
                        n_colour_bottom_2=row["n_colour2_bottom_2"],
                        colour_bottom_2=row["colour2_bottom_2"],
                        pl_bottom=pl2_bottom,
                        shape_bottom_2=row["shape_bottom_2"]
                    )
                elif row["n_colour2_bottom_2"] == "0":
                    if row["shape_bottom_2"] == "cross":
                        pl1_bottom = "es"
                    else:
                        pl1_bottom = "s"
                    if row["n_colour1_top_2"] == "1":
                        pl1_top = ""
                        if row["shape_top_2"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                    elif row["n_colour2_top_2"] == "1":
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                    else:
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                    message2 = template2_3.format(
                        n_colour1_top_2=row["n_colour1_top_2"],
                        colour1_top_2=row["colour1_top_2"],
                        pl1_top=pl1_top,
                        n_colour2_top_2=row["n_colour2_top_2"],
                        colour2_top_2=row["colour2_top_2"],
                        pl2_top=pl2_top,
                        shape_top_2=row["shape_top_2"],
                        n_colour_bottom_2=row["n_colour1_bottom_2"],
                        colour_bottom_2=row["colour1_bottom_2"],
                        pl_bottom=pl1_bottom,
                        shape_bottom_2=row["shape_bottom_2"]
                    )
                else:
                    if row["n_colour1_top_2"] == "1":
                        pl1_top = ""
                        if row["shape_top_2"] == "cross":
                            pl2_top = "es"
                        else:
                            pl2_top = "s"
                        if row["n_colour1_bottom_2"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_2"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_2"] == "1":
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    elif row["n_colour2_top_2"] == "1":
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                        else:
                            pl1_top = "s"
                        pl2_top = ""
                        if row["n_colour1_bottom_2"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_2"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_2"] == "1":
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    else:
                        if row["shape_top_2"] == "cross":
                            pl1_top = "es"
                            pl2_top = "es"
                        else:
                            pl1_top = "s"
                            pl2_top = "s"
                        if row["n_colour1_bottom_2"] == "1":
                            pl1_bottom = ""
                            if row["shape_bottom_2"] == "cross":
                                pl2_bottom = "es"
                            else:
                                pl2_bottom = "s"
                        elif row["n_colour2_bottom_2"] == "1":
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                            else:
                                pl1_bottom = "s"
                            pl2_bottom = ""
                        else:
                            if row["shape_bottom_2"] == "cross":
                                pl1_bottom = "es"
                                pl2_bottom = "es"
                            else:
                                pl1_bottom = "s"
                                pl2_bottom = "s"
                    message2 = template2_1.format(
                        n_colour1_top_2=row["n_colour1_top_2"],
                        colour1_top_2=row["colour1_top_2"],
                        pl1_top=pl1_top,
                        n_colour2_top_2=row["n_colour2_top_2"],
                        colour2_top_2=row["colour2_top_2"],
                        pl2_top=pl2_top,
                        shape_top_2=row["shape_top_2"],
                        n_colour1_bottom_2=row["n_colour1_bottom_2"],
                        colour1_bottom_2=row["colour1_bottom_2"],
                        pl1_bottom=pl1_bottom,
                        n_colour2_bottom_2=row["n_colour2_bottom_2"],
                        colour2_bottom_2=row["colour2_bottom_2"],
                        pl2_bottom=pl2_bottom,
                        shape_bottom_2=row["shape_bottom_2"]
                    )
                # Sentence
                message3 = template3.format(
                    sentence=row["sentence"]
                )
                # Answer options
                option = get_shuffled_options(rng)
                message4 = template4.format(
                    option1=option[0],
                    option2=option[1]
                )
                
                item = "Currently the following pair of pictures is presented:\n" + message1 + message2 + message3 + message4 + "Your answer: I choose\n"
                experiment_list.append([item, row["trial_type"], row["correct_card"], row["set_id"]])
                number += 1
                
    except FileNotFoundError:
        print("File Not Found")

    except KeyError as e:
        print(f"row {e} is not found")
    
    return experiment_list


def make_prompts(item_list, instruction):
    prompts = []
    
    fewshot_filled = instruction.format(
        fewshot_1 = item_list[0][0] + item_list[0][2],
        fewshot_2 = item_list[1][0] + item_list[1][2],
        fewshot_3 = item_list[2][0] + item_list[2][2],
        fewshot_4 = item_list[3][0] + item_list[3][2],
        fewshot_5 = item_list[4][0] + item_list[4][2],
        fewshot_6 = item_list[5][0] + item_list[5][2],
        fewshot_7 = item_list[6][0] + item_list[6][2],
        fewshot_8 = item_list[7][0] + item_list[7][2]
    )
    prompts.append([fewshot_filled+item_list[8][0], "filler", item_list[8][2], item_list[8][3]])
    for item in item_list[9:]:
        prompts.append(item)
    return prompts


def check_correctness(llm_response, trial_type, correct_card):
    resp = llm_response.strip().lower()
    corr = correct_card.strip().lower()
    
    # correct: 1, false: 0
    if trial_type == "target":
        if resp == "first":
            pred = 1
        else:
            pred = 0
    else:
        if resp == corr:
            pred = 1
        else:
            pred = 0

    return pred