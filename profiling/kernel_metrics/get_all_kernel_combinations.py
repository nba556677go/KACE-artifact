#load every csv file in kernel_profiles/output into one dataframe
import os
import pandas as pd
import sys

#error handle args
if len(sys.argv) < 4:
    print("please provide the kernel directory , num_combinations and output prefix")
    print("python get_all_combinations.py kernel_dir num_combinations output_prefix")
    exit(1)

KERNEL_DIR = sys.argv[1]
N_COMB = int(sys.argv[2])
#KACE
#KERNEL_DIR = '../analysis/kernel_profiles/output_ncu'
#hotcloud
#KERNEL_DIR = '/Users/bing/Documents/mlProfiler/tests/mps/analysis/baselines/hotcloud/kernel_hotcloud'
SELECTED_BATCH_SIZE = [2,4,8]
OUTPUT_PREFIX = sys.argv[3]


def get_all_kernels():
    df_features = pd.DataFrame()
    for file in os.listdir(KERNEL_DIR):
        if file.endswith('.csv'):
            # Extract the batch number from the filename
            parts = file.split('_')
            batch_part = next((part for part in parts if part.startswith('batch')), None)
            if batch_part:
                # Extract the numeric part from 'batchX'
                try:
                    batch_size = int(''.join(filter(str.isdigit, batch_part)))
                except ValueError:
                    continue
                if batch_size not in SELECTED_BATCH_SIZE:
                    continue
                df_temp = pd.read_csv(f'{KERNEL_DIR}/{file}')
                df_features = pd.concat([df_features, df_temp], ignore_index=True)
    return df_features

df_features = get_all_kernels()

#label the Type column  in df_feature in labels based on index
labels = {}
for i, label in enumerate(df_features['Type'].unique()):
    labels[label] = i
#df_features['Type'] = df_features.index.map(labels)
print(labels)
# Generate combinations of the dictionary keys
from itertools import combinations_with_replacement
key_combinations = combinations_with_replacement(labels.keys(), N_COMB)
#types, idx = [], []

data = []
#print(key_combinations)
# Print each combination and their corresponding values
for w_combinations in key_combinations:
    combined_data = []
    combined_values = {}
    
    for idx, w in enumerate(w_combinations):
        value = labels[w]
        df_w = df_features[df_features['Type'] == w]
        
        # Drop 'Type' column and add prefix
        df_w = df_w.drop(['Type'], axis=1)
        df_w = df_w.add_prefix(f"w{idx+1}_")
        
        # Convert to dictionary and add to combined data
        combined_data.append(df_w.to_dict(orient='records')[0])
        combined_values[f"workload{idx+1}"] = w
        combined_values[f"idx{idx+1}"] = value
    
    # Combine all the dictionaries into one
    final_data = {**combined_values}
    for d in combined_data:
        final_data.update(d)
    
    data.append(final_data)

# Create a DataFrame from the list
df = pd.DataFrame(data)

# Print the DataFrame
print(df.head())

#print(len(types),types)
#print(len(idx),idx)



#create a new dataframe that has columns = [workload1, workload2]
# rows = types[0] types[1]...
# Create a DataFrame from the list
df = pd.DataFrame(data)

# Print the DataFrame
df.head()
#wrte df to csv
str_batches = "-".join(map(str, SELECTED_BATCH_SIZE))
df.to_csv(f'{OUTPUT_PREFIX}_kernel_labels_comb{N_COMB}_batches{str_batches}.csv', header=True, index=False)

#get the sum of both df_w1 and df_w2

df_features