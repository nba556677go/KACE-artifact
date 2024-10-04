

###CONSTANTS############################################################################################
#CSV_FILE='/home/cc/mlProfiler/tests/mps/multiinstance/kernel_labels_comb3_batches2-8.csv'
#CSV_FILE='/home/cc/mlProfiler/tests/mps/multiinstance/shuffled_comb4_batch2-8output.csv'
CSV_FILE='/home/cc/mlProfiler/tests/mps/multiinstance/shuffled_exclude_oom_kernel_labels_comb4_fixedbatch4_with_batches2-8.csv'

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

while IFS=, read -r workload1 workload2 workload3 workload4; do
    # Skip the header
    if [[ "$workload1" == "workload1" ]]; then
        continue
    fi
    
    echo "Processing row:"
    echo "workload1: $workload1"
    echo "workload2: $workload2"
    echo "workload3: $workload3"
    echo "workload4: $workload4"
    
    # Parse workloads
    parsed_w1=$(parse_workload $workload1)
    parsed_w2=$(parse_workload $workload2)
    parsed_w3=$(parse_workload $workload3)
    parsed_w4=$(parse_workload $workload4)
    
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

    workload_name4=$(echo $parsed_w4 | cut -d, -f1)
    batchsize4=$(echo $parsed_w4 | cut -d, -f2)
    mode4=$(echo $parsed_w4 | cut -d, -f3)
    prefix4=$(echo $parsed_w4 | cut -d, -f4)
    task4=$(echo $parsed_w4 | cut -d, -f5)

    echo "workload1: workload_name=$workload_name1, batchsize=$batchsize1, mode=$mode1, prefix=$prefix1, task=$task1"
    echo "workload2: workload_name=$workload_name2, batchsize=$batchsize2, mode=$mode2, prefix=$prefix2, task=$task2"
    echo "workload3: workload_name=$workload_name3, batchsize=$batchsize3, mode=$mode3, prefix=$prefix3, task=$task3"
    echo "workload4: workload_name=$workload_name4, batchsize=$batchsize4, mode=$mode4, prefix=$prefix4, task=$task4"

    echo "Starting Docker instances for colocation: $workload_name1, $workload_name2, $workload_name3, $workload_name4"

    CURR_LOG_DIR="$LOG_DIR/${workload_name1}_batch${batchsize1}-${mode1}%${workload_name2}_batch${batchsize2}-${mode2}%${workload_name3}_batch${batchsize3}-${mode3}%${workload_name4}_batch${batchsize4}-${mode4}"
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

    if [ "$mode4" == "inf" ]; then
        mode4="inference"
    fi
    
    
    #return
    #exit 0
    for RUN in $(seq 1 $RUNS); do 
        echo RUN$RUN...

        # Start Docker instance for workload1
        docker run --rm --name w1 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/torch:/root/.cache/torch \
        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
        nba556677/ml_tasks:latest \
        /bin/sh -c "cd /root/mlprofiler/workloads/${mode1}; python ${task1}-${mode1}.py --model_name ${prefix1}${workload_name1} \
        --batch_size ${batchsize1} --log_dir ../../tests/mps/multiinstance/$CURR_LOG_DIR --profile_nstep 1000"  \
        > $CURR_LOG_DIR/w1_${workload_name1}_batch${batchsize1}-${mode1}_MPS100.log 2>&1 & pid1=$!

        # Start Docker instance for workload2
        docker run --rm --name w2 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/torch:/root/.cache/torch \
        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
        nba556677/ml_tasks:latest \
        /bin/sh -c "cd /root/mlprofiler/workloads/${mode2}; python ${task2}-${mode2}.py --model_name ${prefix2}${workload_name2} \
        --batch_size ${batchsize2} --log_dir ../../tests/mps/multiinstance/$CURR_LOG_DIR  --profile_nstep 1000"  \
        > $CURR_LOG_DIR/w2_${workload_name2}_batch${batchsize2}-${mode2}_MPS100.log 2>&1 & pid2=$!

        # Start Docker instance for workload3
        docker run --rm --name w3 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/torch:/root/.cache/torch \
        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
        nba556677/ml_tasks:latest \
        /bin/sh -c "cd /root/mlprofiler/workloads/${mode3}; python ${task3}-${mode3}.py --model_name ${prefix3}${workload_name3} \
         --batch_size ${batchsize3} --log_dir ../../tests/mps/multiinstance/$CURR_LOG_DIR  --profile_nstep 1000"  \
        > $CURR_LOG_DIR/w3_${workload_name3}_batch${batchsize3}-${mode3}_MPS100.log 2>&1 & pid3=$!

        #start Docker instance for workload4
        docker run --rm --name w4 --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 \
        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -v ~/.cache/torch:/root/.cache/torch \
        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
        nba556677/ml_tasks:latest \
        /bin/sh -c "cd /root/mlprofiler/workloads/${mode4}; python ${task4}-${mode4}.py --model_name ${prefix4}${workload_name4} \
         --batch_size ${batchsize4} --log_dir ../../tests/mps/multiinstance/$CURR_LOG_DIR  --profile_nstep 1000"  \
        > $CURR_LOG_DIR/w4_${workload_name4}_batch${batchsize4}-${mode4}_MPS100.log 2>&1 & pid4=$!
        # Wait for any Docker instances to complete
        wait -n $pid1 $pid2 $pid3 $pid4
        echo "Processes have finished. Cleaning up..."
        sleep 1

        # Terminate clients
        docker kill w1 w2 w3 w4
        bash ../terminate_client.sh
        sudo kill -9 $(pgrep -f "docker run")
        sudo kill -9 $(pgrep -f "nvidia-smi pmon")
        
        wait $pid1 $pid2 $pid3 $pid4 2>/dev/null
        sleep 5
    done
    echo "-------"
done < <(tail -n +2 $CSV_FILE | cut -d, -f1,3,5,7)  # Skip the header and read the first three columns for workloads