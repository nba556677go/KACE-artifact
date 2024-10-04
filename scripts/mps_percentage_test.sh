

#bash mps_percentage_test.sh rtx6000_logs/sharetest IAT/0301/0301_100reqs_lambd2.0.json 2>rtx6000_logs/sharetest/err.log 1>rtx6000_logs/sharetest/stdout.log
LOG_DIR=$1
arrival_file=$2
# add $3 as one of two modes as train or inference. store in BE_MODE add help message as well
BE_MODE=$3
#BE_MODE should be train or inference only
if [ "$BE_MODE" != "train" ] && [ "$BE_MODE" != "inference" ]; then
    echo "BE_MODE should be train or inference only"
    exit 1
fi

TASKS=("recommend" "imgclassification")
MODELS=("bert-base-cased" "microsoft/resnet-50")
#task:model:epoch:batch_size

#INFERENCE
#task:model:epoch:batch_size
#TASKS_MODELS=("imgclassification:microsoft/resnet-50:0:32" "recommend:bert-base-cased:0:8" "imgclassification:google/mobilenet_v2_1.0_224:0:64" "imgclassification:google/vit-base-patch16-224:0:64" "speech-recognition:facebook/wav2vec2-base-960h:0:1" "speech-recognition:openai/whisper-large-v2:0:1")
TASKS_MODELS=("imgclassification:microsoft/resnet-50:0:32" "recommend:bert-base-cased:0:8" "imgclassification:google/mobilenet_v2_1.0_224:0:64" "speech-recognition:openai/whisper-large-v2:0:1")
#BE_EPOCHS=(7 7) 
#LS_PERCENTAGES=(10 20 30 40 50 60 70 80 90)
#BE_PERCENTAGES=(90 80 70 60 50 40 30 20 10)
LS_PERCENTAGES=(30 50 70 90)
BE_PERCENTAGES=(70 50 30 10)
#LS_PERCENTAGES=(0)
#BE_PERCENTAGES=(100)
RUNS=1


mkdir -p $LOG_DIR
for RUN in $(seq 1 $RUNS); do 
    echo RUN$RUN...;
    # Loop through each LS_PERCENTAGE and BE_PERCENTAGE pair
    for ((i=0; i<${#LS_PERCENTAGES[@]}; i++)); do
        LS_PERCENTAGE=${LS_PERCENTAGES[$i]}
        BE_PERCENTAGE=${BE_PERCENTAGES[$i]}
        echo "LS percent=$LS_PERCENTAGE, BE percent=$BE_PERCENTAGE"
        
        # Loop through each TASK
        for TASK_MODEL in "${TASKS_MODELS[@]}"; do
            IFS=':' read -r TASK MODEL BE_EPOCH BE_BATCH <<< "$TASK_MODEL"
            echo "TASK=$TASK, MODEL=$MODEL BE_EPOCH=$BE_EPOCH BATCH=$BE_BATCH"
            CURR_LOG_DIR="$LOG_DIR/RUN${RUN}/LS${LS_PERCENTAGE}/${TASK}/${MODEL}"
            echo "CURR_LOG_DIR=$CURR_LOG_DIR"
            mkdir -p "$CURR_LOG_DIR"
            
            # Start LS server only when LS_PERCENTAGE is not 0
            if [ $LS_PERCENTAGE -eq 0 ]; then
                echo "LS_PERCENTAGE is 0. Skipping LS server..."
            else
                # Start LS server
                echo "Starting LS server..."
                docker run --rm --name LS --runtime nvidia --gpus all \
                -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/mlProfiler:/root/mlprofiler -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$LS_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
                --env "HUGGING_FACE_HUB_TOKEN=<access_token>" \
                --cap-add=SYS_ADMIN --ipc=host -p 8000:8000 \
                vllm/vllm-openai:latest --model mistralai/Mistral-7B-Instruct-v0.2 \
                --gpu-memory-utilization 0.65 --max-model-len 2000 --dtype=half &
                
                # Wait for LS server to warm up
                echo "Warming up LS server..."
                sleep 60

                # Start sending LS requests
                echo "Start sending LS requests from $arrival_file..."
                python poisson_arrival.py --ip 127.0.0.1 -l 2 -n 500 -f $arrival_file \
                > "$CURR_LOG_DIR/arrival_${TASK}_LSMPS${LS_PERCENTAGE}.log" 2>&1 &
                
                # Log metrics for LS server
                python vllm_logging.py $CURR_LOG_DIR > "$CURR_LOG_DIR/LSmetric_${TASK}_MPS${LS_PERCENTAGE}.log" 2>&1 &
            fi

            
            

            
            #BEMODE = train
            if [ "$BE_MODE" == "train" ]; then
                echo "BE_MODE=train"
                # Start BE job
                docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                nba556677/ml_tasks:latest\
                /bin/sh -c "cd /root/mlprofiler/workloads/training; python ${TASK}-train.py --n_epoch ${BE_EPOCH} \
                --model_name ${MODEL} --batch_size ${BE_BATCH}" \
                > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1
            else
                echo "BE_MODE=inference"
                # Start BE job
                docker run --rm --name BE --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$BE_PERCENTAGE \
                --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 \
                -v /tmp/nvidia-mps:/tmp/nvidia-mps \
                -v ~/.cache/huggingface:/root/.cache/huggingface \
                -v ~/.cache/torch:/root/.cache/torch \
                -v ~/mlProfiler:/root/mlprofiler --ipc=host \
                nba556677/ml_tasks:latest \
                /bin/sh -c "cd /root/mlprofiler/workloads/inference; python ${TASK}-inference.py --model_name ${MODEL} \
                --batch_size ${BE_BATCH} --log_dir ../../tests/mps/$CURR_LOG_DIR"  \
                > $CURR_LOG_DIR/BE_${TASK}_MPS${BE_PERCENTAGE}.log 2>&1
            fi
            
            
            echo "BE job finishes. cleaning up poisson..."
            sudo kill -9 $(pgrep -f "python poisson")
            sudo kill -15 $(pgrep -f "python vllm_logging.py")
            sleep 1
            # Terminate client
            bash terminate_client.sh
            sudo kill -9 $(pgrep -f "python3 -m")
            sudo kill -9 $(pgrep -f "docker run")
            sleep 5
        done
    done
done