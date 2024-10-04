#!/bin/bash

# List of workloads
workloads=(
    'whisper-large-v2_batch16-inf' 'whisper-large-v2_batch8-inf'
    'whisper-large-v2_batch2-inf' 'bert-base-cased_batch16-inf'
    'bert-base-cased_batch8-inf' 'vit-base-patch16-224_batch8-inf'
    'vit-base-patch16-224_batch2-inf' 'vit-base-patch16-224_batch16-inf'
    'wav2vec2-base-960h_batch2-inf' 'wav2vec2-base-960h_batch16-inf'
    'wav2vec2-base-960h_batch8-inf' 'bert-base-cased_batch2-inf'
    'vit_h_14_batch8-train' 'vit_h_14_batch16-train'
    'bert-base-cased_batch16-train' 'bert-base-cased_batch8-train'
    'vit_h_14_batch2-train' 'albert-base-v2_batch2-train'
    'albert-base-v2_batch8-train' 'albert-base-v2_batch16-train'
    'bert-base-cased_batch2-train'
)

# Path to the dataset
#dataset_path="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/MPS100/MPS100_kernel_labels_targetMPS100.csv"
dataset_path="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/total_labels_batch2-8_targetMPS100.csv"
# Output directory
#output_dir="/Users/bing/Documents/mlProfiler/output/dataset/MPS100/partition_exp/n_occur_0/rand10/KACE"
output_dir="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/batch2-8/MPS100/unseen_partition"
# Python script path
python_script="/Users/bing/Documents/mlProfiler/output/dataset/MPS100/partition_exp/partition_data_by_workloadname.py"

#n_combinations
n_combination="3"

# Loop through each workload and call the Python script
for workload in "${workloads[@]}"
do
    echo "Partitioning dataset for workload: $workload"
    mkdir -p $output_dir/$workload
    python $python_script $dataset_path $workload $output_dir/$workload $n_combination
done

echo "Partitioning complete."