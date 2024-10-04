   
import os
import sys
import time
import argparse
import evaluate
import numpy as np

from transformers import TrainingArguments, Trainer
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm

#import custom functions
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger


def inference(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    logger = MLLogger(args.log_dir, args, __file__)
    #log all args
    logger.log(f"args={args}")
    logger.log(f"using {args.device}...")

        
    """
    TODO Count data download & process time
    """
    dataset = load_dataset(args.data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.remove_columns(["text"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    #small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(30000))
    
    #train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    model.to(args.device)


    def countAverageProcessingTime(writecsv=True):
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")
        #write req_processing_time to a csv file
        if writecsv:
            logger.writecsv(f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv", [avg_processing_time])
    

    model.eval()
    req_processing_time, req_total_time = [], 0
    warmup = 10
    steps = 0
    #add batch num in eval_dataloader for loop
    for i, batch in enumerate(eval_dataloader):
        if steps >= warmup:  
            if args.profile:
                torch.cuda.nvtx.range_push(f"steps{steps}")
        start_time = time.time()
        if args.profile  :
            torch.cuda.nvtx.range_push("moveData")
        batch = {k: v.to(args.device) for k, v in batch.items()}
        if args.profile:
            torch.cuda.nvtx.range_pop()
        
        if steps == warmup :
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
        if steps >= warmup:
            if args.profile:
                torch.cuda.nvtx.range_push(f"forward{steps}")   
        with torch.no_grad():
            outputs = model(**batch)
        if steps >= warmup:
            if args.profile:
                torch.cuda.nvtx.range_pop()
        logits = outputs.logits
        if args.profile_1step:
            print("in 1step")
            exit(0)
        
        predictions = torch.argmax(logits, dim=-1)
        print(predictions)
        
        #append process time to req process time
        req_processing_time.extend([time.time() - start_time] * args.batch_size)
        req_total_time += time.time() - start_time
        logger.log(f"request processing time: {req_processing_time[-1]} seconds")
        countAverageProcessingTime(writecsv=False)
        if args.profile_nstep > 0 and i >= args.profile_nstep:
            if args.profile:
                torch.cuda.cudart().cudaProfilerStop()
            countAverageProcessingTime()
            logger.log(f"completed {i} steps, exiting...")
            exit(0)
        if args.profile:
            torch.cuda.nvtx.range_pop()
        steps += 1
        
    avg_processing_time = np.mean(req_processing_time)
    logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
    logger.log(f"Total processing time: {req_total_time:.4f} seconds")
    #write req_processing_time to a csv file
    logger.writecsv(f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv", req_processing_time)
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    #args.model llm - "lmsys/vicuna-7b-v1.5"
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)
