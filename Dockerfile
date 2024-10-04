FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel
WORKDIR /app
RUN pip install datasets transformers numpy evaluate accelerate scikit-learn tqdm pathlib wandb matplotlib pandas termcolor

#
ARG API_KEY
# Set environment variables
ENV WANDB_API_KEY=${API_KEY} \
    WANDB_SILENT=true
# Run the script to perform the copy operation
COPY template ./template
COPY utils ./utils
COPY workloads ./workloads
COPY models ./models
WORKDIR /app/template


#TODO Run Dockerfile and see if the command runs
#docker run [IMAGE] python [train_file] --n_epoch 20 --profile --profile_all_estimation
