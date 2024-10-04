"""
TODO model - vgg, resnet, mobilenet, Alexnet
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from datasets import load_dataset
import torchvision.models as models
from torchvision.transforms import transforms
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModelForImageClassification
import evaluate
#import custom modules
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger
from models.CNN import CNNModel
import signal


def inference(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    
    logger = MLLogger(args.log_dir, args, __file__)
    
    # Load the dataset
    testing_dataset = load_dataset("cifar10", split="test")
    logger.log(f"Dataset loaded. dataset length: {len(testing_dataset)}")
    # Load ResNet-50 model and img processor to convert images to tensor
    model = AutoModelForImageClassification.from_pretrained(args.model_name)
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    if "cuda" in str(args.device):
        model.to(args.device)



    

    def countAverageProcessingTime():
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")
        #write req_processing_time to a csv file
        logger.writecsv(f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv", [avg_processing_time])

    #define sigterm handler that exectute countAverageProcessingTime()
    def sigterm_handler(signum, frame):
        countAverageProcessingTime()
        logger.log("completed nstep, exiting...")
        exit(0)
    #handle sigterm signal
    signal.signal(signal.SIGTERM, sigterm_handler)

    model.eval()
    req_processing_time, req_total_time = [], 0
    warmup = 10
    steps = 0
    # inference the testing dataset
    for i in range(0, len(testing_dataset), args.batch_size):
        if steps < warmup:
            logger.log(f"warmup steps {steps}")
        if steps == warmup :
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
        if steps >= warmup:  
            if args.profile:
                torch.cuda.nvtx.range_push(f"steps{steps}")

        logger.log(f"Processing data = data {i} to {i + args.batch_size}")
        if steps >= warmup:  
            if args.profile  :
                torch.cuda.nvtx.range_push(f"moveData{steps}")
        batch = testing_dataset[i:i + args.batch_size]
        images = processor(batch["img"], return_tensors="pt")
        #if "cuda" in str(args.device):
        images= images["pixel_values"].to(args.device)
        if steps >= warmup:  
            if args.profile  :
                torch.cuda.nvtx.range_pop()

        
        start_time = time.time()
            

        if steps >= warmup:  
            if args.profile  :
                torch.cuda.nvtx.range_push(f"forward{steps}")        
        with torch.no_grad():
            outputs = model(images)
        if steps >= warmup:  
            if args.profile  :
                #pop forward range
                torch.cuda.nvtx.range_pop()
    
        if args.profile_1step:
            logger.log("in 1step")
            exit(0)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        logger.log(f"predictions: {predictions}")
        if steps >= warmup:  
            if args.profile  :
                #pop steps range
                torch.cuda.nvtx.range_pop()
            #append process time to req process time
        req_processing_time.extend([time.time() - start_time] * args.batch_size)
        req_total_time += time.time() - start_time
        #logger.log(f"request processing time: {req_processing_time[-1]} seconds")

        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {req_total_time:.4f} seconds")

        if args.profile_nstep > 0 and i >= args.batch_size * args.profile_nstep:
            if steps >= warmup:  
                if args.profile  :
                    torch.cuda.cudart().cudaProfilerStop()
            countAverageProcessingTime()
            logger.log(f"completed {i}, exiting...")
            exit(0)
        
        steps += 1
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    avg_processing_time = np.mean(req_processing_time)
    logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
    logger.log(f"Total processing time: {req_total_time:.4f} seconds")
    #write req_processing_time to a csv file
    logger.writecsv(f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv", req_processing_time)
    


if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)
    exit(0)
