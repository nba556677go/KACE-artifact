

# run vllm server
```

docker run --rm  --name LS --runtime nvidia -v /usr/local/cuda:/nsys -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/mlProfiler:/root/mlprofiler --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100  --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 --env "HUGGING_FACE_HUB_TOKEN=<access_token>" --cap-add=SYS_ADMIN --ipc=host -p 8000:8000  /nsys/bin/nsys profile -t cuda,nvtx vllm/vllm-openai:latest --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.65 --max-model-len 2000  --dtype=half
```
# run Best effort docker 

```
docker run --rm --name BE -it -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/.cache/torch:/root/.cache/torch --env CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 --env NVIDIA_VISIBLE_DEVICES=0 --gpus device=0 -v /tmp/nvidia-mps:/tmp/nvidia-mps -v ~/mlProfiler:/root/mlprofiler --ipc=host nba556677/ml_tasks:latest bash
```
# entrypoint command
```
# inference
cd /root/mlprofiler/workloads/inference/ && python imgclassification-inference.py --model_name microsoft/resnet-50 --batch_size 64  --log_dir $log_dir

cd /root/mlprofiler/workloads/inference/ &&  python speech-recognition-inference.py --model_name facebook/wav2vec2-base-960h --log_dir ../../tests/mps/tmp --batch_size 2
```

# run Best Effort job
```
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=100 python imgclassification-train.py --model_name microsoft/resnet-50 --batch_size 8 --n_epoch 10 > img_25_PERCENTAGE.log

```


