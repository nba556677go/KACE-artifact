

###CONSTANTS############################################################################################
CSV_FILE='/home/cc/mlProfiler/tests/mps/multiinstance/kernel_labels_comb3_batches2.csv'

#!/bin/bash
# Array of workload names
workload_array=('wav2vec2-base-960h' 'bert-base-cased' 'mobilenet' 'mobilenet_v2_1.0_224' 'vit-base-patch16-224' 'vit_h_14' 'whisper-large-v2' 'vit-base-patch16-224' 'albert-base-v2')
#workloadprefix dict to map workload name to add prefix of the workload file
declare -A workloadprefix
workloadprefix['wav2vec2-base-960h']='facebook/'
workloadprefix['bert-base-cased']=''
workloadprefix['mobilenet']=''
workloadprefix['mobilenet_v2_1.0_224']='google/'
workloadprefix['vit-base-patch16-224']='google/'
workloadprefix['vit_h_14']=''
workloadprefix['whisper-large-v2']='openai/'
workloadprefix['vit-base-patch16-224']='google/'
workloadprefix['albert-base-v2']=''


#declare task_model dict to map workload name to task and model
declare -A task_model
task_model['wav2vec2-base-960h']='speech-recognition'
task_model['bert-base-cased']='recommend'
task_model['mobilenet']='imgclassification'
task_model['mobilenet_v2_1.0_224']='imgclassification'
task_model['vit-base-patch16-224']='imgclassification'
task_model['vit_h_14']='imgclassification'
task_model['whisper-large-v2']='speech-recognition'
task_model['vit-base-patch16-224']='imgclassification'
task_model['albert-base-v2']='recommend'


############################################################################################################
LOG_DIR=$1
RUNS=1

# Function to parse a workload string and extract workload_name, batchsize, and mode
parse_workload() {
    local workload=$1
    local workload_name=""
    local batchsize=""
    local mode=""
    local prefix=""
    local task=""

    # Extract mode (either '-train' or '-inf')
    if [[ $workload =~ -(train|inf)$ ]]; then
        mode=${BASH_REMATCH[1]}
    fi

    # Extract batch size
    if [[ $workload =~ _batch([0-9]+) ]]; then
        batchsize=${BASH_REMATCH[1]}
    fi

    # Extract workload name by removing the batch size and mode parts
    workload_name=${workload/_batch${batchsize}-${mode}/}

    # Get prefix and task
    prefix=${workloadprefix[$workload_name]}
    task=${task_model[$workload_name]}

    echo "$workload_name,$batchsize,$mode,$prefix,$task"
}

while IFS=, read -r workload1 workload2 workload3; do
    # Skip the header
    if [[ "$workload1" == "workload1" ]]; then
        continue
    fi
    
    echo "Processing row:"
    echo "workload1: $workload1"
    echo "workload2: $workload2"
    echo "workload3: $workload3"
    
    # Parse workloads
    parsed_w1=$(parse_workload $workload1)
    parsed_w2=$(parse_workload $workload2)
    parsed_w3=$(parse_workload $workload3)
    
    workload_name1=$(echo $parsed_w1 | cut -d, -f1)
    batchsize1=$(echo $parsed_w1 | cut -d, -f2)
    mode1=$(echo $parsed_w1 | cut -d, -f3)
    prefix1=$(echo $parsed_w1 | cut -d, -f4)
    task1=$(echo $parsed_w1 | cut -d, -f5)

    workload_name2=$(echo $parsed_w2 | cut -d, -f1)
    batchsize2=$(echo $parsed_w2 | cut -d, -f2)
    mode2=$(echo $parsed_w2 | cut -d, -f3)
    prefix2=$(echo $parsed_w2 | cut -d, -f4)
    task2=$(echo $parsed_w2 | cut -d, -f5)

    workload_name3=$(echo $parsed_w3 | cut -d, -f1)
    batchsize3=$(echo $parsed_w3 | cut -d, -f2)
    mode3=$(echo $parsed_w3 | cut -d, -f3)
    prefix3=$(echo $parsed_w3 | cut -d, -f4)
    task3=$(echo $parsed_w3 | cut -d, -f5)

    echo "workload1: workload_name=$workload_name1, batchsize=$batchsize1, mode=$mode1, prefix=$prefix1, task=$task1"
    echo "workload2: workload_name=$workload_name2, batchsize=$batchsize2, mode=$mode2, prefix=$prefix2, task=$task2"
    echo "workload3: workload_name=$workload_name3, batchsize=$batchsize3, mode=$mode3, prefix=$prefix3, task=$task3"

    echo "Starting Docker instances for colocation: $workload_name1, $workload_name2, $workload_name3"

    CURR_LOG_DIR="$LOG_DIR/${workload_name1}_bacth${batchsize1}-${mode1}%${workload_name2}_batch${batchsize2}-${mode2}%${workload_name3}_batch${batchsize3}-${mode3}"
    mkdir -p "$CURR_LOG_DIR"
    #if mode is inf, change mode=inference
    if [ "$mode1" == "inf" ]; then
        mode1="inference"
    fi
    if [ "$mode2" == "inf" ]; then
        mode2="inference"
    fi
    if [ "$mode3" == "inf" ]; then
        mode3="inference"
    fi


    echo "-------"
done < <(tail -n +2 $CSV_FILE | cut -d, -f1,3,5)  # Skip the header and read the first three columns for workloads