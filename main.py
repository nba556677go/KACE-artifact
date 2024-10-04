import argparse
from utils.predictor import Predictor
from utils.trainer import Trainer
from utils.loaddata import getstage1trainData, train_test_split_with_counts, getcloud_stage1trainData, get_multiinstance_stage1trainData
from pathlib import Path

def predict(args, testingfile, model, excluded_cols):
    model = Path(args.model)
    #stage 1
    predictor = Predictor(args = args, 
                        modelpath=model,
                        stage1_data = testingfile,
                        stage2_data="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/stage2/baseline_steps_stage2.csv",
                        actual_share_throughput_data=[args.sharedThroughputData],
                        excluded_cols = excluded_cols)
    if args.predAllacc:
        best_pair = predictor.predictAllWorkloads(HPworkload="", 
                                        target="sum_throughput", 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
        predictor.plot_results(outname=f"{args.output_dir}/allacc_HP-_MPS{args.targetMPS}.png", 
                           pair=best_pair)

        exit(0)
    if args.pred_multiinstance:
        best_pair = predictor.predictAll_multiinstance_Workloads(HPworkload=args.HPworkload, 
                                        target="sum_throughput", 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
        exit(0)
    #stage1
    best_pair = predictor.predictPair(HPworkload=args.HPworkload, 
                                        target="sum_throughput", 
                                        randomseed=30, correlation=args.correlation,
                                        rulebase=args.rulebase)
    
    print(best_pair)
    
        
    
    #stage2
    #best_thread = predictor.getBestThread(policy="max-min", pair=best_pair)
    

    #plot results
    strbest_pair = "-".join(best_pair)
    predictor.plot_results(outname=f"{args.output_dir}/HP-{args.HPworkload}_MPS{args.targetMPS}.png", 
                           pair=best_pair)


def train(args):
    print("start training...")
    trainer = Trainer(args)
    excluded_cols = trainer.processData(data=args.train_file, correlation=args.correlation)
    #selected_feats=["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio"]

    
    modelfile = trainer.train(modeltype=args.modeltype)
    print(f"model saved to {modelfile}")
    #save modelpath to 
    return modelfile, excluded_cols

def getstage1Data(args):
    if args.modeltype == "hotcloud":
        """
        datafile = getcloud_stage1trainData(baselineData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_baseline_labels.csv",
                                            shareThroughputData=args.sharedThroughputData,
                                            kernelData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_kernel_labels.csv",
                                            outname="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_labels.csv",
                                            targetMPS=args.targetMPS,
                                            selected_feats=["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"],
                                            n_combination=args.n_combination)
        """
        datafile = get_multiinstance_stage1trainData(baselineData="/Users/bing/Documents/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_baseline_labels.csv", 
                       shareThroughputData=args.sharedThroughputData, 
                       kernelData="/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/hotcloud_kernel_labels_comb4_batches2-8.csv",
                       outname=f"/Users/bing/Documents/mlProfiler/tests/mps/multiinstance/dataset/hotcloud/0907hotcloud_total_labels_comb{args.n_combination}_batch2-8", 
                       targetMPS=args.targetMPS,
                       n_combination=args.n_combination,
                       hotcloud=True)
    
    else:
        if args.pred_multiinstance:
            datafile = get_multiinstance_stage1trainData(baselineData="./tests/mps/analysis/baseline_labels.csv", 
                       shareThroughputData=args.sharedThroughputData, 
                       kernelData="./tests/mps/multiinstance/shuffled_exclude_oom_kernel_labels_comb4_fixedbatch4_with_batches2-8.csv",
                       outname=f"./tests/mps/multiinstance/dataset/{args.modeltype}/0909_total_labels_comb{args.n_combination}_testbatch4_batch2-8", 
                       targetMPS=args.targetMPS,
                       n_combination=args.n_combination)
        else:
            datafile = getstage1trainData(baselineData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baseline_labels.csv", 
                        shareThroughputData=args.sharedThroughputData, 
                        kernelData="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/kernel_labels.csv",
                        outname="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/MPS100_kernel_labels", 
                        targetMPS=args.targetMPS)
    
    
    return datafile
    


if __name__ == "__main__":
    #stage1Data = getstage1Data()
    #argparse
    #add argparse


    parser = argparse.ArgumentParser()

    #common
    parser.add_argument("--output_dir","-o", type=Path, help="output directory of training model and predictions") 
    #for predictions
    parser.add_argument("--model", "-m",type=Path)
    parser.add_argument("--HPworkload", "-w", type=str)
    parser.add_argument("--targetMPS", "-t", type=int)
    

    #for split data
    parser.add_argument("--processStage1Data", "-pd", action="store_true")
    parser.add_argument("--customsplit","-c", action="store_true")
    parser.add_argument("--nonSplitData", "-d",type=Path)
    parser.add_argument("--split_randomseed","-rd", type=int, default=30)
    parser.add_argument("--n_occur", "-n", type=int, default=4)
    parser.add_argument("--sharedThroughputData", "-sd", type=Path)
    

    #data
    #train file "output/dataset/training_set.csv"
    parser.add_argument("--train_file", type=Path)
    #test file,="output/dataset/testing_set.csv"
    parser.add_argument("--test_file", type=Path)
    parser.add_argument("--correlation","-corr", type=float, default=0.2)
    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument('--custom_col_exclude', '-excol', type=list_of_strings, default=[])

    #train
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--modeltype", "-mt", type=str, choices=["hotcloud", "KACE", "AutoML", "RF", "NN"])

    #baselines
    parser.add_argument("--rulebase", "-rb", action="store_true")

    #noHPworkloads - get mse/r2 for all workloads in testing set without specifying HPworkload
    parser.add_argument("--predAllacc", "-ga", action="store_true")

    #multuinstance
    parser.add_argument("--n_combination", "-comb", type=int, default=2)
    parser.add_argument("--pred_multiinstance", "-predmulti", action="store_true")
    args = parser.parse_args()


    print("in main...")
    if args.train:  
        model, excluded_cols = train(args) 
        print("training done. exit...")
        
        exit(0)

    #process stage1Data from mergeing kernels, shareddata, and baselines
    if args.processStage1Data:
        stage1Data = getstage1Data(args) 
        exit(0)
#split data if provided
    if args.customsplit:
        trainset, testset = train_test_split_with_counts(data=Path(args.nonSplitData),  
                                    train_outname=args.train_file, 
                                    test_outname=args.test_file, n_workload_counts=args.n_occur, random_seed=args.split_randomseed)
    
        exit(0)
    #train, testfile should be provided together to delete low-correlation columns
    excluded_cols = []
    if args.train_file:
        print(f"process training data at {args.train_file}")
        #need excluded columns to pass to predictor for process test file...
        excluded_cols = Trainer(args).processData(data=args.train_file, correlation=args.correlation)
        print(f"training file processed. excluded cols={excluded_cols}")
    


    testfile = Path(args.test_file) if args.test_file else  Path(args.nonSplitData)

    #predict - need to have sharedThroughputData
    predict(args, testfile, args.model,  excluded_cols)


    
    """
    #process stage1Data
    if args.processStage1Data:
        stage1Data = getstage1Data(args) 
        
    if args.customsplit:
        
        trainset, testset = train_test_split_with_counts(data=Path(args.stage1Data),  
                                     train_outname=args.training_file, 
                                     test_outname=args.testing_file, n_workload_counts=3)
        stage1Data = testset
            
    else:
        #load stage1Data from existed file
        stage1Data = Path(args.stage1Data)

    
    predict(args, model)

    
  """
        
    
    #HPworkloads = ["bert-base-cased_batch2-train","vit-base-patch16-224_batch2-inf", "vit_h_14_batch16-train", "wav2vec2-base-960h_batch8-inf", "bert-base-cased_batch2-inf"]
    
    #for HPworkload in HPworkloads:
    #    predict(HPworkload, stage1Data)
    
    #predict(args, 
    #        targetMPS=args.targetMPS)
    