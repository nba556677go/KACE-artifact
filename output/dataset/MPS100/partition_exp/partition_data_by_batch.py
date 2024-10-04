import pandas as pd
import sys
#argument check
if len(sys.argv) < 7:
    print("please provide the datafile outdir num_combinations batch")
    print("python partition_data_by_batch.py datafile outdir num_combinations batch testfilename trainfilename")
    exit(1)
# Load your dataset
data_file = pd.read_csv(sys.argv[1])
outdir = sys.argv[2]
n_combination = int(sys.argv[3])
batch = sys.argv[4]
testfile = sys.argv[5]
trainfile = sys.argv[6]
# Function to check if all workloads contain 'batch2'
def is_all_batch_num(row, n_combination, batch):
    for i in range(1, n_combination + 1):
        if f'batch{batch}' not in row[f'workload{i}']:
            return False
    return True


# Create a mask where all workloads contain 'batch2'
train_mask = data_file.apply(lambda row: is_all_batch_num(row, n_combination, batch), axis=1)

# Split into training and testing sets
train_set = data_file[train_mask]
test_set = data_file[~train_mask]

# Check the results
print(f"Training set: {len(train_set)} rows")
print(f"Testing set: {len(test_set)} rows")

# Optionally, save the datasets to CSV
train_set.to_csv(f'{outdir}/{trainfile}', index=False)
test_set.to_csv(f'{outdir}/{testfile}', index=False)
