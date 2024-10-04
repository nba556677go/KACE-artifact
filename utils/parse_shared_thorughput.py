#parse share directory
#PARAMETER
#directory: the directory of the share test
#LStype: the type of the LS, train or inf
#BEtype: the type of the BE, train or inf
#LSpercent: the percentage of the LS
import pandas as pd
from collections import defaultdict
import sys
sys.path.append('../utils')
from parse_util import *
import sys


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("please provide the input file output prefix and num of combinations")
        print("python stage2.py inputfile output_predix num_combinations")
        exit(1)
    base_df = pd.read_csv("baseline_steps_stage2.csv")
    share_directory_path = sys.argv[1]
    output_prefix = sys.argv[2]
    N_COMB=int(sys.argv[3])
    #share_directory_path = "/home/cc/mlProfiler/tests/mps/ccv100_logs/share_3work_batch2-8"
    #share_directory_path = "/home/cc/mlProfiler/tests/mps/multiinstance/ccv100/comb4_testbatch4_trainbatch2-8"
    #get all models from basedf workload column
    all_models = base_df["workload"].tolist()
    print(all_models)
    LStype, BEtype = "train", "train"
    share_file_steps = defaultdict(dict)
    #share_directory_path = "/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/cclogs/ccv100_0624_MPS100/sharetest/LStrain_batch2816_BEtrain_batch2816"
    #share_file_steps = get_share_avgStep(share_directory_path, LStype=LStype, BEtype=BEtype, all_models=all_models)
    #save_share_file_steps(share_file_steps=share_file_steps, filename=f"{sys.argv[1]}_{LStype}-{BEtype}_share_steps_stage2.csv")


    #share_directory_path = "/home/cc/mlProfiler/tests/mps/ccv100_logs/share_3work_batch2-8"
    share_file_steps = get_multiworkloads_share_avgStep(share_directory_path, N_COMB, all_models)
    #test_share = {('wav2vec2-base-960h_batch8-inf', 'wav2vec2-base-960h_batch2-inf', 'vit_h_14_batch8-train', 'vit_h_14_batch8-train'): {'w1_wav2vec2-base-960h_batch8-inf_MPS100': 50.87042635254873, 'w2_wav2vec2-base-960h_batch2-inf_MPS100': 19.328484606954266}}
    save_multiinstance_share_file_steps(share_file_steps=share_file_steps, filename=f"{output_prefix}_share_comb{N_COMB}_steps_stage2.csv", n_combination=N_COMB)
 
    
