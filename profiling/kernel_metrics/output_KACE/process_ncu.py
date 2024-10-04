import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, required=True,help='input ncu file path')
parser.add_argument('--results_dir', type=str, required=True,
                        help='output path for processed profiling files')
parser.add_argument('--num_GPUs', type=int, default=4, help='Number of GPUs')
parser.add_argument('--job_type', type=str, help='workload name ex. bert-train_batch2')
args = parser.parse_args()

df = pd.read_csv(f'{args.input_file}')
print(df.head())
kernels = []
metrics_to_get = ['Duration', 'Block Size', 'Grid Size', 'Threads','Compute (SM) Throughput', 'DRAM Throughput', "Memory Throughput", 'Registers Per Thread', 'Static Shared Memory Per Block']

unique_kernel_names = set()
"""
skips - GPU content should be skipped:
Content example:
"GPU ID","GPU Name","NUMA ID by CPU Affinity","CPU Affinity","NUMA ID by Memory Affinity"
"0","Tesla V100-SXM2-32GB","0","0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78","0"
"1","Tesla V100-SXM2-32GB","0","0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78","0"
"2","Tesla V100-SXM2-32GB","1","1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79","1"
"3","Tesla V100-SXM2-32GB","1","1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79","1"
"""
skips_GPU_content = 0
#skips - GPU content should be skipped:
#
for index, row in df.iterrows():
    #skip the next args.num_GPUs+1 rows if GPU content

    if skips_GPU_content > 0:
        print(f"Skipping row {index+2}")
        skips_GPU_content -= 1
        continue
    kernel = row['Kernel Name']
    metric_name = row['Metric Name']
    
    ID = row['ID']
    #if "GPU ID" in first column, skip the next args.num_GPUs+1 rows
    if row[0] == "GPU ID":
        #skip args.num_GPUs+1 rows
       
        print(f"Skipping GPU info rows starting from index {index+2}")
        skips_GPU_content = args.num_GPUs
        continue
   


    #first entry 
    
    
    if metric_name == 'DRAM Frequency' and  str(row['Body Item Label']) == "nan":
        #print(f"index: {index+2}, new kernel: {kernel}")
        kernels.append({"kernel" : kernel, "ID": ID})
        unique_kernel_names.add(kernel)
    elif metric_name in metrics_to_get and  str(row['Body Item Label']) == "nan":
        #SPECIAL handle: memory thoruhgput should only be recorded for the section name of "GPU Speed Of Light Throughput"
        if metric_name == "Memory Throughput" and str(row["Section Name"]) != "GPU Speed Of Light Throughput":
            continue
        
        #print(index+2, metric_name)
        kernels[-1][metric_name] = float( str(row['Metric Value']).replace(',', ''))

for x in unique_kernel_names:
    print(x)
    print("------------------------------------")

print("num of total kernels" , len(kernels))

#compute weighted sum of features by duration
total_duration = 0
total_columns = ["Threads", 'Compute (SM) Throughput', 'DRAM Throughput', "Memory Throughput", "Registers", "Static Shared Memory"]
#init total dict with total columns
total_dict = {key: 0 for key in total_columns}
import math
for kernel in kernels:
    #print(kernel)
    #raise error if any of the required metrics is missing
    for metric in metrics_to_get:
        if metric not in kernel:
            raise ValueError(f"Missing metric {metric} in kernel {kernel['kernel']}")
            exit(1)
    #thread = block size(thread per block) * grid size (block per grid)
    #num_threads = math.floor(kernel["Block Size"]) * math.floor(kernel[-3])
    num_registers = kernel["Threads"] * math.floor(kernel['Registers Per Thread'])
    kernel['Registers'] = num_registers
    kernel['Static Shared Memory']  = kernel['Grid Size'] * kernel['Static Shared Memory Per Block']

    total_duration += kernel['Duration']
    for col in total_columns:
        total_dict[col] += kernel[col] * kernel['Duration']

#compute weighted sum
for col in total_columns:
    total_dict[col] = total_dict[col]/total_duration
total_dict["Type"] = args.job_type
#add  total duration to total dict
#total_dict['Duration'] = total_duration

    


print(len(kernels))
#print(kernels[0])
#labels =['Kernel_Name', 'DRAM_Throughput(%)', 'Duration(ns)', 'Compute(SM)(%)',  'Block', 'Grid', 'Registers_Per_Thread', 'Static_shmem_per_block', 'Number_of_threads', 'Number_of_registers']
labels = kernels[0].keys()


df_new = pd.DataFrame(kernels, columns=labels)
print(df_new)

df_new.to_csv(f'tmp/output_{args.job_type}_ncu_kernels_processed.csv',index=False)
#save total dict to csv file
df_total = pd.DataFrame([total_dict])
df_total.to_csv(f'{args.results_dir}/output_{args.job_type}_ncu_total.csv',index=False)