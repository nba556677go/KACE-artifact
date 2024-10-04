import pandas as pd
import sys

def partition_dataset(filename, test_workload_name, outputdir, n_combination, split_test_only):
    n_combination = int(n_combination)
    # Load the dataset
    df = pd.read_csv(filename)
    
    exact_test_mask = pd.Series([False] * len(df))
    exclude_workload_task_mask = pd.Series([False] * len(df))

    for i in range(1, n_combination+1):
        exact_test_mask = exact_test_mask | (df[f'workload{i}'] == test_workload_name)
        #exclude_workload_task_mask = exclude_workload_task_mask | (df[f'workload{i}'].str.startswith(base_workload_name) & df[f'workload{i}'].str.endswith(f'-{task}'))
    # Create masks for filtering
    #exact_test_mask = (df['workload1'] == test_workload_name) | (df['workload2'] == test_workload_name)
    #exclude_workload_task_mask = (
    #    (df['workload1'].str.startswith(base_workload_name) & df['workload1'].str.endswith(f'-{task}')) |
    #    (df['workload2'].str.startswith(base_workload_name) & df['workload2'].str.endswith(f'-{task}'))
    #)
    
    # Training mask excludes the exact test workload but includes workloads with different tasks
    train_mask = ~exact_test_mask 
    test_mask = exact_test_mask

    # Split the data into training and testing based on the mask
    train_df = df[train_mask]
    test_df = df[test_mask]
    print(f"outputdir: {outputdir}")
    # Save the training and testing datasets
    if split_test_only:
        test_df.to_csv(f'{outputdir}/testing_set.csv', index=False)
        print(f"Testing set saved to {outputdir}/testing_set.csv")
        return
    train_df.to_csv(f'{outputdir}/training_set.csv', index=False)
    print(f"Traing set saved to {outputdir}/training_set.csv")
    #make dir for test_workload_name

    # Print csv file names
    return 

# Usage example:
if __name__ == "__main__":
    print(sys.argv)
    filename = sys.argv[1]
    test_workload_name = sys.argv[2]
    outputdir = sys.argv[3]
    n_combination = sys.argv[4]
    split_test_only = True if len(sys.argv) >= 6 else False
    #print(sys.argv)
    partition_dataset(filename, test_workload_name, f"{outputdir}", n_combination, split_test_only)

