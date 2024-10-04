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
from utils.Dataset import DummyBertDataset


def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    logger = MLLogger(args.log_dir, args, __file__)
    #log all args
    logger.log(f"args={args}")
    logger.log(f"using {args.device}...")
    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            logger.log(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs
        
    """
    TODO Count data download & process time
    """
    
    vocab_size = 30522  
    dummy_dataset = DummyBertDataset(num_samples=10000, seq_length=512, vocab_size=vocab_size)
    
    train_dataloader = DataLoader(dummy_dataset, shuffle=True, batch_size=args.batch_size)
    #eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size)

    """
    end TODO of Counting data download & process time
    """
    if args.ckpt:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.log('check on ckpt model...')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
        if args.load_from_ckpt:
            logger.log(f'loading ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
    model.to(args.device)

    if args.profile:
        logger.log(f'Model parameters = {count_parameters(model)}')
        profiler.getModelMemory()

    optimizer = AdamW(model.parameters(), lr=args.lr)


    num_training_steps = args.n_epoch * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    #import wandb
    if args.wandb_logging:
        import wandb

        #wandb.init(project=f"{os.path.basename(__file__)[:-3]}_epochs_{args.n_epoch}_batch_size_{args.batch_size}_profile_earlystop_{args.profile}_profile_epochs_{args.profile_run}", entity="nba556677go", name=args.exp_name, config=args)
        wandb.init(
        # set the wandb project where this run will be logged
        entity="nba556677go",
        project="k8s-scheduling",
        name=args.exp_name,
        # track hyperparameters and run metadata
        config=args  
        )

        wandb.watch(model)

    progress_bar = tqdm(range(num_training_steps))


    start_time = time.time()
    step_process_time = []
    warmup = 10
    steps =  0
    model.train()
    for epoch in range(1, run_epochs+1):

        start_epoch_time = time.time()
        running_loss = 0.0
        
        for batch in train_dataloader:
            if steps == warmup :
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
            if steps >= warmup:  
                if args.profile:
                    torch.cuda.nvtx.range_push(f"steps{steps}")
            #print(f"batch: {batch.values()[0].shape}"
            step_time = time.time()

            if steps >= warmup:  
                if args.profile:
                    torch.cuda.nvtx.range_push(f"moveData{steps}")
            batch = {k: v.to(args.device) for k, v in batch.items()}
            if steps >= warmup:  
                if args.profile:
                    torch.cuda.nvtx.range_pop()

            
            #create randn tensor to replace batch
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"forward{steps}")   
            outputs = model(**batch)
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            loss = outputs.loss
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"backward{steps}")   
            loss.backward()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"optimize{steps}")   
            optimizer.step()
            optimizer.zero_grad()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"schedule{steps}")   
            lr_scheduler.step()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            
            progress_bar.update(1)

            running_loss += loss.item()
            step_process_time.extend([time.time() - step_time] * args.batch_size)
            #logger.countAverageProcessingTime(step_process_time)
            logger.log(f"average step time: {sum(step_process_time) / len(step_process_time):.4f} seconds")
            if args.profile_1step:
                logger.log(f'profile only 1 step for nightsight compute...')
                exit(0)
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            if args.profile_nstep > 0 and  steps >= args.profile_nstep:
                #logger.countAverageProcessingTime(step_process_time)
                logger.log(f"completed {args.profile_nstep} steps, exiting...")
                torch.cuda.cudart().cudaProfilerStop()
                exit(0)
            steps += 1
            
        
        epoch_time = time.time() - start_epoch_time
        train_log = {
            "train_loss": running_loss / len(train_dataloader),
        }
        logger.log(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
            GPUinfo = profiler.getSMIinfobyTask(sys.argv)
            logger.log(f"[Profile]GPU INFO - {GPUinfo}")
            gpuInfoDict = list(GPUinfo.values())[0][0]
            profile_log = {
                "GPUMem (MB)" : int(gpuInfoDict["gpu_mem"][:-3]),
                "CPUpercentage" : float(gpuInfoDict["cpu"]),
                "RAMpercentage" : float(gpuInfoDict["mem"]),
                "Time elapsed": epoch_time,

            }
            profiler.addEpochTime(epoch_time)
            if args.wandb_logging:
                wandb.log({**profile_log, "epoch": epoch})

        if args.wandb_logging:
            wandb.log({**train_log, "epoch": epoch})


            
        

    total_training_time = time.time() - start_time
    m, s = divmod(total_training_time, 60)
    h, m = divmod(m, 60)
    logger.log(f"Total Training Time: {h:f}h {m:02f}m {s:02f}s")
    training_time_per_epoch = total_training_time / run_epochs
    m, s = divmod(training_time_per_epoch , 60)
    h, m = divmod(m, 60)
    logger.log(f"Average Time for each Epoch: {h:f}h {m:02f}m {s:02f}s")

    if args.wandb_logging:
        wandb.log({"Total Training Time": total_training_time,
                    "Average Time for each Epoch": training_time_per_epoch,
                    })
    if args.profile and not args.profile_all_estimation:
        profiler.saveEarlyStop(os.path.basename(__file__)[:-3], GPUinfo, training_time_per_epoch)
        logger.log(f"Earlystop at epoch{run_epochs}")

    """
    TODO sample for saving check point model - for future preemption purpose 
    """
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt_dir / f"model_{args.device}_{args.exp_name}.pt")
        logger.log(f"{'':30s}*** Best model saved ***")


    ''' 
    Profiling task after training
    '''
    if args.profile and args.profile_all_estimation:
        result_csv = profiler.saveAllEpochEstimatedTime(total_training_time, training_time_per_epoch)
        cwd = os.getcwd()
        profiler.draw_summary(f"{result_csv}")
        logger.log(f'Current working directory is {cwd}')
        logger.log("Profile finished!")
            
    logger.log("Training finished!")


    model.eval()


    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)



if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    #args.model llm - "lmsys/vicuna-7b-v1.5"
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)
