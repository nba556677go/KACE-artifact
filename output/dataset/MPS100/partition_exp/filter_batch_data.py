import pandas as pd

# Load the CSV file into a DataFrame
csv_file = '/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/0906_kernel_labels_comb4_batches2-4-8.csv'  # Replace with your CSV file path
n_combination = 4  # Replace with the number of combinations
batch = 4  # Replace with the batch number
df = pd.read_csv(csv_file)

# Define the columns that contain workload information
workload_columns = [f'workload{i+1}' for i in range(n_combination)]  # Assuming 4 workloads: workload1, workload2, workload3, workload4

# Filter rows where at least one workload contains 'batch2'
filtered_df = df[df[workload_columns].apply(lambda row: any(f'batch{batch}' in str(workload) for workload in row), axis=1)]

# Save the filtered dataframe to a new CSV file (optional)
filtered_df.to_csv(f'filtered_batch{batch}_rows.csv', index=False)

# Print filtered results
print(filtered_df)
