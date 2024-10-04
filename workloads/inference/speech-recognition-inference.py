"""

sample model_names: facebook/wav2vec2-base-960h, openai/whisper-large-v2"
"""


import os
import sys
import time
import argparse
import evaluate
import numpy as np

import datasets
from transformers import pipeline, Wav2Vec2Processor, Wav2Vec2ForCTC,WhisperProcessor, WhisperForConditionalGeneration
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
#import custom modules
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger

#set logging level and timestamp format as datetime

##REAL DATASET- NOT USED
class ASRDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        audio = self.dataset[idx]['audio']['array']
        #print(self.dataset[idx])
        input_values = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True).input_features
        #print(input_values)
        print(input_values.shape)
        return input_values.squeeze(0)

##USED - DUMMY DATASET WITH SAME LENGTH
class DummyDataset(Dataset):
    def __init__(self, num_samples, seq_length, num_channels=0):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.num_channels = num_channels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.num_channels > 0:
            return torch.randn(self.num_channels, self.seq_length)
        else:
            return torch.randn(self.seq_length)

def collate_fn(batch):
    # Pad the batch of audio tensors so they all have the same length
    batch = [item.squeeze(0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch
"""


"""

def calibrate_batched_time(req_processing_time, batch_size):
    
    #req_processing_time is a list of time for each batch, and the batch time should be count as batch[0]'s time + batch[i]
    #example - if batch_size = 4 , req_processingtim[:batch_size] = [3.7,0.0015, 0.0012,0.009] 
    #then the req time should be [3.7, 3.7015, 3.7027, 3.7117]
    if batch_size == 1:
        #direct return sum since not batch is considered
        return sum(req_processing_time)
    
    total_process_time = 0
    for i in range(0, len(req_processing_time), batch_size):
        for j in range(1, batch_size):
            if i+j >= len(req_processing_time):
                total_process_time += req_processing_time[i+j-1]
                break
            req_processing_time[i+j] = req_processing_time[i] + req_processing_time[i+j]
            if j == batch_size - 1:
                total_process_time += req_processing_time[i+j]
    return total_process_time
    #return req_processing_time

    
def inference(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    #create a logger, file dir=args.log_dir + args.model_name combine with os path
    logger = MLLogger(args.log_dir, args, __file__)
    #log all args
    logger.log(f"args={args}")

    # Load the dataset
    

    #dataset = datasets.load_dataset("superb", name="asr", split=f"test[:20%]")
    if  "wav2vec" in args.model_name:
        processor = Wav2Vec2Processor.from_pretrained(args.model_name)
        model = Wav2Vec2ForCTC.from_pretrained(args.model_name).to(args.device)
        dummy_dataset = DummyDataset(num_samples=10000, seq_length=160487, num_channels=0)
    elif "whisper" in args.model_name:
        processor = WhisperProcessor.from_pretrained(args.model_name)
        logger.log(f"loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_name).to(args.device)
        dummy_dataset = DummyDataset(num_samples=10000, seq_length=3000, num_channels=80)
    else:
        logger.log(f"model {args.model_name} not supported")
        exit(1)

  
    logger.log(f"Use dummy dataset -  Inference starts with batch size={args.batch_size} and model={args.model_name} dataset length: {len(dummy_dataset)}")
    #asr_dataset = ASRDataset(dataset, processor)
    
    data_loader = DataLoader(dummy_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    #save length of each input length in a list to csv
    input_lengths = []
    req_processing_time = []
    warmup = 10


    start_time = time.time()
    def countAverageProcessingTime(writecsv=True):
        avg_processing_time = np.mean(req_processing_time)
        logger.log(f"Average processing time: {avg_processing_time:.4f} seconds")
        logger.log(f"Total processing time: {avg_processing_time * len(req_processing_time):.4f} seconds")
        #write req_processing_time to a csv file
        logger.writecsv(f"req_device{args.device}batch{args.batch_size}_inference_process_time.csv", [avg_processing_time])
    
    for steps, batch in enumerate(data_loader):
        if steps == warmup :
            if args.profile:
                torch.cuda.cudart().cudaProfilerStart()
                
        if steps >= warmup:  
            if args.profile:
                
                torch.cuda.nvtx.range_push(f"steps{steps}")
                torch.cuda.nvtx.range_push(f"moveData{steps}")
        start_time = time.time()
        batch = batch.to(args.device)
        if steps >= warmup:  
            if args.profile:
                torch.cuda.nvtx.range_pop()
            #logger.log(f"batch shape: {batch.shape}")
            
            input_lengths.append(batch.shape[1])
            if args.profile  :
                torch.cuda.nvtx.range_push(f"forward{steps}")
            if "whisper" in args.model_name:
                outputs = model.generate(batch)
            else:
                outputs = model(batch).logits
            if args.profile  :
                torch.cuda.nvtx.range_pop()

            # Take the argmax to get the predicted token ids
            predicted_ids = torch.argmax(outputs, dim=-1)
            
            # Decode the token ids to text
            if args.profile :
                torch.cuda.nvtx.range_push(f"decode{steps}")
            transcriptions = processor.batch_decode(predicted_ids)
            if args.profile:
                torch.cuda.nvtx.range_pop()
            print(transcriptions)
            end_time = time.time()
            req_processing_time.extend([end_time - start_time]* args.batch_size)
            countAverageProcessingTime()
            if args.profile:
                #pop steps range
                torch.cuda.nvtx.range_pop()
            
            if args.profile_nstep > 0 and steps >= args.profile_nstep:
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStop()
                logger.log(f"completed  {steps}steps, exiting...")
                logger.writecsv(f"req_device{args.device}_batch{args.batch_size}__inference_process_time.csv", req_processing_time)
                exit(0)

        #WARMUP- do not record time
        else:
            start_time = time.time()
            logger.log(f"warmup step {steps}")
            if "whisper" in args.model_name:
                outputs = model.generate(batch)
            else:
                outputs = model(batch).logits
             # Take the argmax to get the predicted token ids
            predicted_ids = torch.argmax(outputs, dim=-1)
            # Decode the token ids to text
            transcriptions = processor.batch_decode(predicted_ids)
            end_time = time.time() 
            logger.log(f"Average processing time: {end_time-start_time:.4f} seconds")
        #print(len(req_processing_time))
        
        if args.profile_1step:
            print("in 1step")
            exit(0)
        #print(outputs.shape)
    #save input_lengths to a csv file
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()
    logger.writecsv(f"req_device{args.device}_batch{args.batch_size}__input_lengths.csv", input_lengths)
    logger.writecsv(f"req_device{args.device}_batch{args.batch_size}__inference_process_time.csv", req_processing_time)

    #print(f"mean input length: {np.mean(input_lengths)}")
    return
        

    
if __name__ == "__main__":
    args = MLParser(mode="inference").get_args()
    profiler = MLProfiler(args) if args.profile else None
    inference(args, profiler)