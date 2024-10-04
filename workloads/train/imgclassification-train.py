import sys
import time
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

#import custom functions
sys.path.append('../..')
from utils.util import count_parameters
from utils.parser import MLParser
from utils.profiler import MLProfiler
from utils.logger import MLLogger
from utils.Dataset import DummyImageDataset
from models.CNN import CNNModel
import torchvision.models as models

def train(args: argparse.Namespace, profiler: MLProfiler = None) -> None:
    logger = MLLogger(args.log_dir, args, __file__)
    logger.log(f"using {args.device}...")
    run_epochs = args.n_epoch
    if args.profile:
        GPUinfo = []
        logger.log("[Profile] init phase, starting training...")
        profiler.getMemStats(ts=time.time(), state='load_model')
        if not args.profile_all_estimation:
            run_epochs = min(args.n_epoch, args.profile_n_epoch)
            logger.log(f"[Profile] early stop at epoch{run_epochs}")
            profiler.profile_n_epoch = run_epochs

    # Load and preprocess the CIFAR-10 dataset
    if args.model_name == "resnet" or args.model_name == "mobilenet":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


   

    #test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model and move it to the GPU if available
    if args.ckpt:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger.log('check on ckpt model...')
        #TODO actual model should load from pretrain
        
        if args.load_from_ckpt:
            logger.log(f'loading ckpt {args.ckpt}')
            model.load_state_dict(torch.load(args.ckpt))
        else:
            if "resnet" in args.model_name:
                model = models.resnet50(pretrained=True)
                train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 32, 32))
            else:
                model = CNNModel()
                train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 224, 224))
    else:
        logger.log(f"loading model={args.model_name}...")
        if "resnet" in args.model_name:
            model = models.resnet50(pretrained=True)
            train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 32, 32))
        elif "mobilenet" in args.model_name:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT) 
            train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 32, 32))
        else:
            #print(models.__dict__)first_parameter = next(model.parameters())
            
            model = models.__dict__[args.model_name]()
            train_dataset = DummyImageDataset(num_samples=10000, image_shape=(3, 224, 224))
    model.to(args.device)
     #train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    if args.profile:
        logger.log(f"model to cuda...")
        profiler.countModelParameters(model)
        profiler.getModelMemory()
        profiler.getMemStats(ts=time.time(), state='load_model')
    
        
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
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
    
    #start global time
    start_time = time.time()
    step_process_time = []
    warmup = 10
    steps =  0
    for epoch in range(1, run_epochs+1):
        model.train()
        running_loss = 0.0
        start_epoch_time = time.time()  # Start time of the current epoch
        logger.log(f"getting batches within epoch{epoch}...")
        
        for i, (inputs, labels) in enumerate(train_dataloader, 0):
            #record step time
            if steps == warmup :
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
            if steps >= warmup:  
                if args.profile:
                    torch.cuda.nvtx.range_push(f"steps{steps}")
            
            step_time = time.time()
            if args.profile:
                torch.cuda.nvtx.range_push("moveData")
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            if args.profile:
                logger.log(f"batch{i}, batch len = {len(inputs)}")
                logger.log(f"send inputs to device, input size={(inputs.nelement()*inputs.element_size() + labels.nelement()*labels.element_size()) / 1024} KB")
                #profiler.getMemStats(ts=time.time(), state='load_model')
                torch.cuda.nvtx.range_pop()

            optimizer.zero_grad()
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"forward{steps}")   
            outputs = model(inputs)
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            
            #logger.log("after forward propagation",  profiler.getMemStats(ts=time.time(), state='load_model')) if args.profile else None
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"backward{steps}")  
            loss = criterion(outputs, labels)
            loss.backward()
            
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()
            #logger.log("after backward pass",  profiler.getMemStats(ts=time.time(), state='load_model') ) if args.profile else None
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_push(f"optimize{steps}")  
            optimizer.step()
            optimizer.zero_grad()
            #logger.log("after optimizer stepping",  profiler.getMemStats(ts=time.time(), state='load_model') ) if args.profile else  None
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()

            running_loss += loss.item()
            
            #record step time
            step_process_time.extend([time.time() - step_time] * args.batch_size)
            #logger.log(f"batch{i} done in {step_process_time[-1]:.2f} seconds")
            logger.log(f"average step time: {sum(step_process_time) / len(step_process_time):.2f} seconds")
            #logger.countAverageProcessingTime(step_process_time)
            if steps >= warmup:
                if args.profile:
                    torch.cuda.nvtx.range_pop()#pop steps
            if args.profile_nstep > 0 and  i >= args.profile_nstep:
                #logger.countAverageProcessingTime(step_process_time)
                logger.log(f"completed {args.profile_nstep} steps, exiting...")
                torch.cuda.cudart().cudaProfilerStop()
                exit(0)
            """
            if args.profile:
                logger.log("after add running loss",  profiler.getMemStats(ts=time.time(), state='load_model') ) if args.profile else None
                torch.cuda.reset_peak_memory_stats(device=None) if args.profile else None
                torch.cuda.empty_cache()
            """
            steps += 1

        if args.profile:
            #nsys only profiles 1 epoch
            torch.cuda.cudart().cudaProfilerStop()
            
        


        epoch_time = time.time() - start_epoch_time
        logger.log(f"Epoch {epoch}, Loss: {running_loss / len(train_dataloader)}, Time: {epoch_time:.2f} seconds")
        train_log = {
            "train_loss": running_loss / len(train_dataloader),
        }
        if args.profile:
            logger.log(f"batches done in epoch{epoch}..., reset peak memory")
            profiler.getMemStats(ts=time.time(), state='load_model')
            torch.cuda.reset_peak_memory_stats(device=None)

        if args.profile:
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
"""SKIP EVALUATION
    # Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logger.log(f"Test Accuracy: {100 * correct / total}%")
"""

if __name__ == "__main__":
    args = MLParser(mode="train").get_args()
    profiler = MLProfiler(args) if args.profile else None
    train(args, profiler)
    exit(0)
