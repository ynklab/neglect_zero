import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os

def load_all_subjects():

    all_data = []
    target_groups = ['BAS', 'ESQ', 'DIST', 'UPP']
   
    for group in target_groups:
        for subject_id in range(1, 81):

            # You have to change the filename and path below to match your desired input location
            # example: ./results/llama4scout/result_{group}_Subject{subject_id}.csv
            filename = f"./results/gemma3/4b/result_{group}_Subject{subject_id}.csv"
           
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename)

                    unique_subject_id = f"{group}_{subject_id}"
                   
                    df['Subject'] = unique_subject_id
                    df['Original_ID'] = subject_id
                    df['Group'] = group
                    df['trial_index'] = df.index / len(df)
                   
                    all_data.append(df)
                    print(f"Loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            else:
                print(f"File not found: {filename} - Skipping.")
                pass

    if not all_data:
        raise ValueError("cannot load any valid data files. Please check the paths and filenames.")

    return pd.concat(all_data, ignore_index=True)


def classify_row(row):
    """
    Function to extract necessary information from the Condition string for each row
    """
    cond = row['Condition']
   
    row_type = 'Other'
    if 'filler' in cond:
        row_type = 'Filler'
    elif 'target' in cond:
        row_type = 'Target'
    elif 'prime' in cond:
        row_type = 'Prime'
       
    prime_cat = np.nan
    if cond == 'bas_prime':
        prime_cat = 'BAS'
    elif cond == 'control_prime':
        prime_cat = 'Control'
    elif cond == 'critical_prime':
        prime_cat = 'Critical'
       
    return pd.Series([row_type, prime_cat], index=['Type', 'Prime_Category'])


def main():

    try:
        df_all = load_all_subjects()
    except ValueError as e:
        print(e)
        return

    print("\nProcessing data conditions...")
    df_all[['Type', 'Prime_Category']] = df_all.apply(classify_row, axis=1)

    df_all['Prev_Prime_Cat'] = df_all['Prime_Category'].shift(1)
   
    # You should chage the filename and path below to match your desired output location
    # example: ./results/llama4scout/integrated_result_llama4scout.csv
    output_filename = "results/llama4scout/integrated_result_llama4scout.csv"
    print(f"\nExporting integrated data to {output_filename}...")
    try:
        df_all.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(" -> Done.")
    except Exception as e:
        print(f" -> Failed to export CSV: {e}")


if __name__ == "__main__":
    main()