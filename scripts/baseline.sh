

#bash mps_percentage_test.sh rtx6000_logs/sharetest IAT/0301/0301_100reqs_lambd2.0.json 2>rtx6000_logs/sharetest/err.log 1>rtx6000_logs/sharetest/stdout.log
LOG_DIR=$1
arrival_file=$2
# add $3 as one of two modes as train or inference. store in BE_MODE add help message as well
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



#task:model:epoch:batch_size

#INFERENCE
#task:model:epoch:batch_size
#LS_MODELS=("speech-recognition:facebook/wav2vec2-base-960h:0:1" "speech-recognition:openai/whisper-large-v2:0:1" "imgclassification:google/vit-base-patch16-224:0:1")
LS_MODELS=("speech-recognition:openai/whisper-large-v2:0:1")

#inference
#TASKS_MODELS=("speech-recognition:facebook/wav2vec2-base-960h:0:1" "speech-recognition:openai/whisper-large-v2:0:1" "imgclassification:google/vit-base-patch16-224:0:1" "imgclassification:microsoft/resnet-50:0:32" "recommend:bert-base-cased:0:8" "imgclassification:google/mobilenet_v2_1.0_224:0:64")

#train
TASKS_MODELS=("imgclassification:vit_h_14:2:8" "recommend:albert/albert-base-v2:2:8" "recommend:bert-base-cased:2:8" "imgclassification:mobilenet:2:64" "imgclassification:microsoft/resnet-50:2:32")
#TASKS_MODELS=("imgclassification:google/mobilenet_v2_1.0_224:0:64")


TASKS_MODELS=()
batch_sizes=(2)
train_candidates=(
    "recommend:bert-base-cased"
    "imgclassification:mobilenet"
    "imgclassification:vit_h_14"
    "recommend:albert/albert-base-v2"
    "imgclassification:resnet-50"
)

inference_candidates=(
    "speech-recognition:openai/whisper-large-v2"
    "imgclassification:google/mobilenet_v2_1.0_224"
    "recommend:bert-base-cased"
    "imgclassification:google/vit-base-patch16-224"
    "imgclassification:microsoft/resnet-50"
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

#trainings
#TASKS_MODELS=("imgclassification:vit_h_14:2:8" "recommend:albert/albert-base-v2:2:8")
#TASKS_MODELS=("recommend:albert/albert-base-v2:1:8")

#BE_EPOCHS=(7 7) 
#LS_PERCENTAGES=(0 10 20 30 40 50 60 70 80 90)
#LS_PERCENTAGES=(0 0 0 0 0 0 0)
#BE_PERCENTAGES=(100 90 80 70 60 50 40 30 20 10)
#LS_PERCENTAGES=(0 0 0 0 0 0 0 0 0 0)
#BE_PERCENTAGES=(100 50 90 80 70 60 40 30 20 10)
LS_PERCENTAGES=(0)
BE_PERCENTAGES=(100)
RUNS=1


mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do 
    echo RUN$RUN...;
    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair
    for LS_MODEL in "${LS_MODELS[@]}"; do
        IFS=':' read -r LS_TASK LS_MODEL LS_EPOCH LS_BATCH <<< "$LS_MODEL"
        echo "LS_TASK=$LS_TASK, LS_MODEL=$LS_MODEL LS_EPOCH=$LS_EPOCH LS_BATCH=$LS_BATCH"
        for ((i=0; i<${#LS_PERCENTAGES[@]}; i++)); do
            LS_PERCENTAGE=${LS_PERCENTAGES[$i]}
            BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
            echo "LS percent=$LS_PERCENTAGE, BE percent=$BE_PERCENTAGE"
            
            # Loop through each TASK
            #loop through LS_MODELS as outer loop

            for TASK_MODEL in "${TASKS_MODELS[@]}"; do
                IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
                echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
                CURR_LOG_DIR="$LOG_DIR/RUN${RUN}/LS${LS_PERCENTAGE}/${LS_TASK}/${LS_MODEL}_batch${LS_BATCH}/BE_${MODEL}_batch${BE_BATCH}"
                echo "CURR_LOG_DIR=$CURR_LOG_DIR"
                mkdir -p "$CURR_LOG_DIR"

                # Start LS server only when LS_PERCENTAGE is not 0
                if [ $LS_PERCENTAGE -eq 0 ]; then
                    echo "LS_PERCENTAGE is 0. Skipping LS server..."
                else
                    # Start LS server
                    echo "Starting LS server..."
                    if [ "$LS_MODE" == "train" ]; then
                        docker run --rm --name LS --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:latest \
                        /bin/sh -c "cd /root/mlprofiler/workloads/train; python ${LS_TASK}-train.py --model_name ${LS_MODEL} --n_epoch ${LS_EPOCH}\
                        --batch_size ${LS_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR"  \
                        > $CURR_LOG_DIR/LS_${LS_TASK}_MPS${LS_PERCENTAGE}.log 2>&1 & ls_pid=$!
                    else
                        docker run --rm --name LS --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
                        --env NVIDIA_VISIBLE_DEVICES=1 --gpus device=1 \
                        -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                        -v ~/.cache/huggingface:/root/.cache/huggingface \
                        -v ~/.cache/torch:/root/.cache/torch \
                        -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                        --cap-add=SYS_ADMIN -v /opt/nvidia/nsight-systems/2023.3.3:/nsys \
                        nba556677/ml_tasks:latest \
                        /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${LS_TASK}-inference.py --model_name ${LS_MODEL} \
                        --batch_size ${LS_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR --profile_nstep 200"  \
                        > $CURR_LOG_DIR/LS_${LS_TASK}_MPS${LS_PERCENTAGE}.log 2>&1 & ls_pid=$!
                    fi
                fi

                

                
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
                    /bin/sh -c "cd /root/mlprofiler/workloads/train; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                    --model_name ${MODEL} --batch_size ${BE_BATCH} --profile_nstep 100" \
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
                    --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR --profile_nstep 100"  \
                    > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1 & be_pid=$!
                fi
                #nvidia-smi pmon -o DT -i 1 -f $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}_gpu_info.csv &
                nvidia-smi pmon -o DT -i 1 > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}_gpu_info.csv &
                # Wait for any process to finish and kill the other
                wait -n $ls_pid $be_pid
                echo "BE process has finished. Cleaning up..."
                sleep 1
                
                # Terminate client
                docker kill LS BE
                bash terminate_client.sh
                sudo kill -9 $(pgrep -f "docker run")
                sudo kill -9 $(pgrep -f "nvidia-smi pmon")
               
                wait $ls_pid $be_pid 2>/dev/null
                sleep 5
            done
        done
    done
done