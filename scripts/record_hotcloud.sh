#!/bin/bash

# The command to run
BE_PERCENTAGE=100
BE_BATCH=2
TASK="recommend"
MODEL="bert-base-cased"

mkdir -p $CURR_LOG_DIR

LOG_DIR=$1
arrival_file=$2
LS_MODE=$3
BE_MODE=$4
if [ "$LS_MODE" != "train" ] && [ "$LS_MODE" != "inference" ]; then
    echo "LS_MODE should be train or inference only"
    exit 1
fi
#BE_MODE should be train or inference only
if [ "$BE_MODE" != "train" ] && [ "$BE_MODE" != "inference" ]; then
    echo "BE_MODE should be train or inference only"
    exit 1
fi


TASKS_MODELS=()
batch_sizes=(2 8 16)
train_candidates=(
    "recommend:bert-base-cased"
    #"imgclassification:mobilenet"
    "imgclassification:vit_h_14"
    "recommend:albert/albert-base-v2"
    #"imgclassification:resnet-50"
)

inference_candidates=(
    "speech-recognition:openai/whisper-large-v2"
    #"imgclassification:google/mobilenet_v2_1.0_224"
    "recommend:bert-base-cased"
    "imgclassification:google/vit-base-patch16-224"
    #"imgclassification:microsoft/resnet-50"
    "speech-recognition:facebook/wav2vec2-base-960h"
)

if [ "$BE_MODE" == "train" ]; then
    epoch=2
    # Loop over tasks and models strings
    for task_model in "${train_candidates[@]}"; do
        # Split the string into task and model using IFS (Internal Field Separator)
        IFS=":" read -r task model <<< "$task_model"
        for batch_size in "${batch_sizes[@]}"; do
            # Append the task:model:epoch:batch_size to the LS_MODELS array
            TASKS_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
        done
    done
fi

if [ "$BE_MODE" == "inference" ]; then
    epoch=0
    # Loop over tasks and models strings
    for task_model in "${inference_candidates[@]}"; do
        # Split the string into task and model using IFS (Internal Field Separator)
        IFS=":" read -r task model <<< "$task_model"
        for batch_size in "${batch_sizes[@]}"; do
            # Append the task:model:epoch:batch_size to the LS_MODELS array
            TASKS_MODELS+=("${task}:${model}:${epoch}:${batch_size}")
        done
    done
fi

LS_PERCENTAGES=(0)
BE_PERCENTAGES=(100)
RUNS=1
#COMMAND="cd /home/cc/mlProfiler/workloads/training && sudo /home/cc/miniconda3/bin/python recommend-train.py --n_epoch 10 --model_name bert-base-cased --batch_size 2 --profile_nstep 20000"
mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do 
    echo RUN$RUN...;
    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair

    for ((i=0; i<${#BE_PERCENTAGES[@]}; i++)); do
        BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
        echo "BE percent=$BE_PERCENTAGE"
        
        # Loop through each TASK
        #loop through LS_MODELS as outer loop

        for TASK_MODEL in "${TASKS_MODELS[@]}"; do
            IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
            echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
            CURR_LOG_DIR="$LOG_DIR/RUN${RUN}/${TASK}/BE_${MODEL}_batch${BE_BATCH}"
            echo "CURR_LOG_DIR=$CURR_LOG_DIR"
            mkdir -p "$CURR_LOG_DIR"

            

            if [ $BE_PERCENTAGE -eq 0 ]; then
                echo "BE_PERCENTAGE is 0. Skipping BE workload..."
            else
                #BEMODE = train
                if [ "$BE_MODE" == "train" ]; then
                    echo "BE_MODE=train"
                    # Start BE job
                    docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                    --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                    -v ~/.cache/huggingface:/root/.cache/huggingface \
                    -v ~/.cache/torch:/root/.cache/torch \
                    -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                    --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                    nba556677/ml_tasks:latest\
                    /bin/sh -c "cd /root/mlprofiler/workloads/training; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                    --model_name ${MODEL} --batch_size ${BE_BATCH} --profile_nstep 500" \
                    > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1  & be_pid=$!
                else
                    echo "BE_MODE=inference"
                    # Start BE job
                    docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                    --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                    -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                    -v ~/.cache/huggingface:/root/.cache/huggingface \
                    -v ~/.cache/torch:/root/.cache/torch \
                    -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                    --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                    nba556677/ml_tasks:latest \
                    /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${TASK}-inference.py --model_name ${MODEL} \
                    --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR --profile_nstep 400"  \
                    > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1 & be_pid=$!
                fi
            fi

            echo  "PID: $be_pid"
            #wait for start ups
            sleep 2
            # Check if the PID was found
            if [ -z "$be_pid" ]; then
              echo "Training script process not found."
              exit 1
            fi
            # Give it a moment to start
            #get docker stats
            CONTAINER_ID=$(docker ps -q --filter "name=BE")
            #while true monitor
            monitor_stats() {
              while true; do
                TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
                docker stats $CONTAINER_ID --no-stream --format "{{ json . }}" | jq --arg timestamp "$TIMESTAMP" '. + {timestamp: $timestamp}' >> $CURR_LOG_DIR/docker_stats.log
                sleep 1
              done
            }

            # Find the PID of the actual Python process running the training script
            #PID=$(pgrep -f "python")

            #remove json file if exists
            rm -f $CURR_LOG_DIR/docker_stats.log

            # Monitor the CPU and memory usage of the process
            # Start monitoring in the background
            monitor_stats & monitor_pid=$!

            # Wait for the Docker container to finish
            wait -n "$be_pid"
            docker kill BE
            echo "ml process complete,clean up..."
            sudo kill -9 $(pgrep -f "docker run")
            sudo kill -9 $(pgrep -f "docker stats")
            sudo kill -9 $monitor_pid
            sleep 5
        done
    done
    
done
# Run the command in the background
#eval $COMMAND &
#get COMMAND PID
#ctrl c handle
#trap "docker kill BE" INT
#PID=$!

