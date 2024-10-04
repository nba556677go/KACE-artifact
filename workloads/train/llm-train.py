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
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.dataloader import get_sample_data_loader
from utils.deepspeed import *

deepspeed_config_dict = {
    "train_batch_size": 1,
    "steps_per_print": 1,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.00015,
        }
    },
    "fp16": {
        "enabled": False,
        "initial_scale_power": 15
    },

    "zero_optimization": {
        "stage": 0,
        "sub_group_size": 8,
        "reduce_bucket_size": 20,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": False,
            "ratio": 0.5
        }
    },
     "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },

}



def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    print(f"using {args.device}...")
    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        profiler.preRunMemStats()
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            print(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs
        
    """
    TODO Count data download & process time
    """
    #dataset = load_dataset(args.data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    
    """
    end TODO of Counting data download & process time
    """
    #model creation
    if args.ckpt:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        print('check on ckpt model...')
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)
        if args.load_from_ckpt:
            print(f'loading ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5)




    #deepspeed
    if args.deepspeed:
        import deepspeed
        #deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_cold(model)
        args = get_ds_args("/tmp/", deepspeed_config_dict, args)
        model, _, _, _ = deepspeed.initialize(args=args,
                                            model=model,
                                            model_parameters=model.parameters(),
                                           )
    if not args.deepspeed:
        model.to(args.device)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        #creating lr_scheduler
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
    
    #create dataloader
    input_dim = next(model.parameters()).size()[0]
    train_dataloader, eval_dataloader = get_sample_data_loader(model=model, total_samples=1000, input_dim=input_dim, device=args.device, batch_size=args.batch_size)
    
    if args.profile:
        profiler.getModelMemory()

    


    num_training_steps = args.n_epoch * len(train_dataloader)
    
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

    model.train()
    for epoch in range(1, run_epochs+1):

        start_epoch_time = time.time()
        running_loss = 0.0
        
        for n, batch in enumerate(train_dataloader):
            #batch = {k: v.to(args.device) for k, v in batch.items()}
            if args.deepspeed:
                #forward() method
                #print(batch.shape)
                print("batch[0]", batch[0])
                print("batch[1]", batch[1].to(torch.long)  )
                print("batch[2]", batch[2])
                print(batch.size())
                loss = model(**batch)
                if dist.get_rank() == 0:
                    print("LOSS:", loss.item())
                #runs backpropagation
                model.backward(loss)

                #weight update
                model.step()
                print_params('step={}'.format(n), model)
            else:
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step() 
                optimizer.zero_grad()   
                progress_bar.update(1)

                running_loss += loss.item()
            

        
        epoch_time = time.time() - start_epoch_time
        train_log = {
            "train_loss": running_loss / len(train_dataloader),
        }
        print(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")
        if args.profile:
            GPUinfo = profiler.getSMIinfobyTask(sys.argv)
            print(f"[Profile]GPU INFO - {GPUinfo}")
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
    print(f"Total Training Time: {h:f}h {m:02f}m {s:02f}s")
    training_time_per_epoch = total_training_time / run_epochs
    m, s = divmod(training_time_per_epoch , 60)
    h, m = divmod(m, 60)
    print(f"Average Time for each Epoch: {h:f}h {m:02f}m {s:02f}s")

    if args.wandb_logging:
        wandb.log({"Total Training Time": total_training_time,
                    "Average Time for each Epoch": training_time_per_epoch,
                    })
    if args.profile and not args.profile_all_estimation:
        profiler.saveEarlyStop(os.path.basename(__file__)[:-3], GPUinfo, training_time_per_epoch)
        print(f"Earlystop at epoch{run_epochs}")

    """
    TODO sample for saving check point model - for future preemption purpose 
    """
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt_dir / f"model_{args.device}_{args.exp_name}.pt")
        print(f"{'':30s}*** Best model saved ***")


    ''' 
    Profiling task after training
    '''
    if args.profile and args.profile_all_estimation:
        result_csv = profiler.saveAllEpochEstimatedTime(total_training_time, training_time_per_epoch)
        cwd = os.getcwd()
        profiler.draw_summary(f"{result_csv}")
        print(f'Current working directory is {cwd}')
        print("Profile finished!")
            
    print("Training finished!")


    metric = evaluate.load("accuracy")
    model.eval()


    for batch in eval_dataloader:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    #args.model llm - "lmsys/vicuna-7b-v1.5"
    profiler = MLProfiler(args) if args.profile else None
    
    if not args.deepspeed:
        print("this script is for deepspeed only. Enable with --deepspeed...")
        exit()

    train(args, profiler)
