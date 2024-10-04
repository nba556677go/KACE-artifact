import pandas as pd
import sys
import os
from collections import defaultdict

def calculate_average_sm_memory(filename):
    # Read the log file into a DataFrame, skipping the header and footer
    df = pd.read_csv(filename, delim_whitespace=True)
    #drop rows with df['sm'] == '-' or df['mem'] == '%'
    #drop rows with df['sm']  not a number
    #drop first 15 rows if total rows > 15
    #drop last 15 rows if total rows > 15
    if len(df) > 10:
        df = df[10:]
    print(df['sm'].unique())
    for col in ['sm', 'mem']:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df.dropna(subset=[col], inplace=True)
    print(df['sm'].unique())
    
    # Calculate the average of 'sm' and 'mem' columns
    avg_sm = df['sm'].astype(float).sum() / len(df)
    avg_mem = df['mem'].astype(float).sum() / len(df)
    return avg_sm, avg_mem

def read_average_sm_memory(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Get the average sm and memory values
    avg_sm = df['average_sm%'].values[0]
    avg_mem = df['average_memory%'].values[0]
    
    return avg_sm, avg_mem
def save_to_csv(avg_sm, avg_mem, filename):
    # Create a DataFrame with the averages
    df_avg = pd.DataFrame({'sm%': [avg_sm], 'mem%': [avg_mem]})

    # Save the DataFrame to a new CSV file
    df_avg.to_csv(filename[:-4] + '_smi.csv', index=False)
    print(f"Average SM% and memory% saved to {filename[:-4] + '_smi.csv'}")


def convert_mib_to_gb(memory_used_mib):
    # Convert MiB to GB and return it as float with 1 decimal precision
    return round(float(memory_used_mib.replace(' MiB', '')) / 1024, 1)

def parse_memory_used_to_gb(filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    
    # Assuming "memory.used [MiB]" is the column to convert
    column_name = [col for col in df.columns if 'memory.used' in col.lower()][0]
    
    # Convert MiB to GB and return the maximum value
    gb_values = df[column_name].apply(convert_mib_to_gb)    
    # Return the list of memory used in GB
    return gb_values.max()


import re
from datetime import datetime

# Define the log file path

def parse_train_log_avgStep_file(filename):
    # Lists to store timestamps and average step times
    timestamps = []
    average_step_times = []

    # Define a regex pattern to match the relevant lines in the log file
    pattern = re.compile(r'\[logger.py:\d+\] (\d+-\d+-\d+ \d+:\d+:\d+,\d+) - INFO - average step time: (\d+\.\d+) seconds')

    # Read the log file and extract the required information
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp_str, avg_step_time_str = match.groups()
                #print(f"timestamp_str: {timestamp_str}, avg_step_time_str: {avg_step_time_str}")
                timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'))
                average_step_times.append(float(avg_step_time_str))

    if len(average_step_times) <= 1  or len(timestamps) <= 1:
            return None
    # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(timestamps) > 10:
        timestamps = timestamps[9:]
    if len(average_step_times) > 10:
         average_step_times = average_step_times[9:]
    # Calculate the total time elapsed
    if timestamps:
        #print(f"start time: {timestamps[0]}")
        #print(f"end time: {timestamps[-1]}")

        time_elapsed = (timestamps[-1] - timestamps[0]).total_seconds()
        
    # Calculate the average step time
    if average_step_times:
        number_of_steps = len(average_step_times)-1#intervals
        average_step_time = number_of_steps / time_elapsed
        print(f'Number of steps: {number_of_steps}')
        print(f'Time elapsed: {time_elapsed} seconds')
        print(f'Average step time: {average_step_time:.4f} steps/second')
        print(f"average time for each step: {1/average_step_time:.4f} seconds/step")
        

    # Save the timestamps and average step times to lists (if needed)
    #print(f'Timestamps: {timestamps}')
    #print(f'Average Step Times: {average_step_times}')
    #print(f"entries {len(timestamps)}, {len(average_step_times)}")
    return  average_step_time

def parse_inf_log_avgStep_file(filename):
    # Lists to store timestamps and average step times
    timestamps = []
    average_step_times = []

    # Define a regex pattern to match the relevant lines in the log file
    pattern = re.compile(r'\[logger.py:\d+\] (\d+-\d+-\d+ \d+:\d+:\d+,\d+) - INFO - Average processing time: (\d+\.\d+) seconds')

    # Read the log file and extract the required information
    with open(filename, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                timestamp_str, avg_step_time_str = match.groups()
                timestamps.append(datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f'))
                average_step_times.append(float(avg_step_time_str))

    # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(average_step_times) <= 1  or len(timestamps) <= 1:
        return None
    
        # exclude first 10 steps in timesteps and average_step_times to exclude slow start
    if len(timestamps) > 10:
        timestamps = timestamps[9:]
    if len(average_step_times) > 10:
         average_step_times = average_step_times[9:]

    # Calculate the total time elapsed
    if timestamps:
        time_elapsed = (timestamps[-1] - timestamps[0]).total_seconds()

    # Calculate the average step time
    if average_step_times:
        number_of_steps = len(average_step_times)-1
        average_step_time = number_of_steps / time_elapsed
        #print(f'Number of steps: {number_of_steps}')
        #print(f'Time elapsed: {time_elapsed} seconds')
        print(f'Average step time: {average_step_time:.4f} steps/second')
        print(f"average time for each step: {1/average_step_time:.4f} seconds/step")

    # Save the timestamps and average step times to lists (if needed)
    #print(f'Timestamps: {timestamps}')
    #print(f'Average Step Times: {average_step_times}')
    print(f"entries {len(timestamps)}, {len(average_step_times)}")
    return  average_step_time
from collections import defaultdict
def get_all_base_avgStep(directory, BEtype):
    THREAD_PERCENTAGES = [i for i in  range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in  THREAD_PERCENTAGES]
    LS_dir = os.path.basename(directory)
    file_steps = defaultdict(list)   # List to store the paths of all files
    
    for root, dirs, files in os.walk(directory):

        for file in files:
            if file.startswith('BE'):
                file_path = os.path.abspath(os.path.join(root, file))
                BE_dir = os.path.basename(root)
                #remove "BE" from BE+dir if it exists
                BE_dir = BE_dir.replace("BE_", "")
                print(file)
            #split file with "_" and find if BASELINE_THREADS is in the split
                for MPSpercent in BASELINE_THREADS:

                    if MPSpercent+".log" in file.split("_") and file.startswith('BE') :
                        file_path = os.path.abspath(os.path.join(root, file))
                        batch_size = int(BE_dir.split("_")[-1][5:])
                        print(file_path)
                        print(f"{BE_dir} {MPSpercent}")
                        print(f"batch_size: {batch_size}")
                        #get batch size from BE_dir
                        
                        if BEtype == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            file_steps[f"{BE_dir}-train"].append((int(f"{MPSpercent[3:]}"),  (parse_train_log_avgStep_file(file_path)* batch_size)))
                        else:
                            if parse_inf_log_avgStep_file(file_path) is not None:
                                file_steps[f"{BE_dir}-inf"].append((int(f"{MPSpercent[3:]}"),  (parse_inf_log_avgStep_file(file_path)* batch_size)))
                        break

                #file_paths.append(file_path)
    return file_steps

def get_share_avgStep(directory, LStype, BEtype, all_models):
    THREAD_PERCENTAGES = [i for i in  range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in  THREAD_PERCENTAGES]
    LS_dir = os.path.basename(directory)
    file_steps = defaultdict(dict)   # List to store the paths of all files
    
    for root, dirs, files in os.walk(directory):

        for file in files:
            for MPSpercent in BASELINE_THREADS:
                if MPSpercent+".log" in file.split("_") and (file.startswith('BE') or file.startswith('LS')):
                    file_path = os.path.abspath(os.path.join(root, file))
                    #split file_path
                    file_path_split = file_path.split("/")
                    
                    print(file)
                    print(file_path_split)
                    LS_model = None
                    #get LS model name by matching file_path_split with all_models
                    for segments in file_path_split:
                        if segments+"-"+LStype in all_models:
                            LS_model = segments
                            break
                    if LS_model is None:
                        print(f"LS model not found in all models, returning...")
                        return
                    print(LS_model)
                    
                    LS_batch_size = int(LS_model.split("_")[-1][5:])
                    BE_dir = os.path.basename(root)
                    #remove "BE" from BE+dir if it exists
                    BE_dir = BE_dir.replace("BE_", "")
                    print(f"LS BE: {LS_model} {BE_dir}")
                #split file with "_" and find if BASELINE_THREADS is in the split
                

                    
                    BE_batch_size = int(BE_dir.split("_")[-1][5:])
                    print(file_path)
                    print(f"{BE_dir} {MPSpercent}")
                    print(f"batch_size: {BE_batch_size}")
                    #get batch size from BE_dir
                    #get LS logs
                    
                    if  file.startswith('LS') :
                        if LStype == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            LS_steps = parse_train_log_avgStep_file(file_path)*LS_batch_size
                            file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"LS{MPSpercent[3:]}_{LS_model}-{LStype}"] = LS_steps
                        elif LStype == "inf" and parse_inf_log_avgStep_file(file_path) is not None:
                            LS_steps = parse_inf_log_avgStep_file(file_path)*LS_batch_size
                            file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"LS{MPSpercent[3:]}_{LS_model}-{LStype}"] = LS_steps
                    
                    elif file.startswith('BE'):
                        if BEtype == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            BE_steps = parse_train_log_avgStep_file(file_path)*BE_batch_size
                            file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"BE{MPSpercent[3:]}_{BE_dir}-{BEtype}"] = BE_steps
                        elif BEtype == "inf" and parse_inf_log_avgStep_file(file_path) is not None:
                            BE_steps = parse_inf_log_avgStep_file(file_path)*BE_batch_size
                            file_steps[(f"{LS_model}-{LStype}", f"{BE_dir}-{BEtype}")][f"BE{MPSpercent[3:]}_{BE_dir}-{BEtype}"] = BE_steps
           
                            
                    break
    print(file_steps)
    #reshape file_steps to a dictionary with key=({LS_model}-{LStype}, {BE_dir}-{BEtype}) and value = (LS_steps, BE_steps)


                        

                #file_paths.append(file_path)
    return file_steps

def parse_workload(log_file):
    #ex. w2_vit_h_14_batch2-inference_MPS100.log
    # Removing the prefix 'w2_' if present, and removing the suffix '.log'
    log_file = log_file.replace('.log', '')
    idx, workload_name, batchsize, mode, MPSpercent = "", "", "", "", ""
    # Splitting the string to extract batchsize, mode, and MPSpercent
    parts = log_file.split('_')
    for part in parts:
        if part.startswith('batch'):
            batchsize, mode = part.split("-")[0].replace('batch', ''), part.split("-")[1]
            if mode !=  "inference" and  mode != "train": 
                raise ValueError(f"Invalid mode: {mode}")
            if mode == "inference":
                mode = "inf"
        #elif part.endswith('inference') or part.endswith('train'):
        #    mode = 'inf' if part.endswith('inference') else 'train'
        elif part.startswith('MPS'):
            MPSpercent = part.replace('MPS', '')
    
    # Extracting the workload_name
    #workload_name should be the second part of the split(_)

    idx, workload_name = log_file.split('_batch')[0].split("_", 1)

    # Extracting the index
    #idx = log_file.split('_')[0]
    
    
    return idx, workload_name, batchsize, mode, MPSpercent


def get_multiworkloads_share_avgStep(directory, n_combinations, all_models):
    THREAD_PERCENTAGES = [i for i in range(10, 101, 10)]
    BASELINE_THREADS = ["MPS" + str(i) for i in THREAD_PERCENTAGES]
    file_steps = defaultdict(dict)  # Dictionary to store the paths of all files

    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            if "%" in subdir:
                workloads = subdir.split('%')
                if len(workloads) != n_combinations:
                    raise ValueError(f"Expected {n_combinations} workloads, but found {len(workloads)} workloads")
                
                #parsed_workloads = [parse_workload(i+1, workload) for i, workload in enumerate(workloads)]
                #print(parsed_workloads)
                subdir_path = os.path.abspath(os.path.join(root, subdir))
                        #continue
                #print(f"subdir contain: {os.listdir(subdir_path)}")
                for file in os.listdir(subdir_path):
                    if ".log" in file:
                    #for idx, workload_name, batchsize, mode in parsed_workloads:
   
                        file_path = os.path.abspath(os.path.join(root,subdir, file))
                        #print(f"file split: {file.split("_")}")
                        idx, workload_name, batchsize, mode, MPSpercent = parse_workload(file)    
                        #print(idx, workload_name, batchsize, mode, MPSpercent)

                        if mode == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            train_steps = parse_train_log_avgStep_file(file_path) * int(batchsize)
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = train_steps
                        elif mode == "inf" and parse_inf_log_avgStep_file(file_path) is not None:
                            inf_steps = parse_inf_log_avgStep_file(file_path) * int(batchsize)
                            file_steps[tuple(workloads)][f"{idx}_{workload_name}_batch{batchsize}-{mode}_MPS{MPSpercent}"] = inf_steps
                            #file_steps[(f"{LS_model}-{mode}", f"{BE_dir}-{mode}")][f"LS{MPSpercent[3:]}_{LS_model}-{mode}"] = LS_steps
                    
                    
                        
                        
    print(file_steps)
    #reshape file_steps to a dictionary with key=({LS_model}-{LStype}, {BE_dir}-{BEtype}) and value = (LS_steps, BE_steps)


                        

                #file_paths.append(file_path)
    return file_steps

def save_multiinstance_share_file_steps(share_file_steps,filename, n_combination):
    import csv
    import itertools
    # Define the set of possible values for x, y, and z
    possible_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Create a list to store the valid combinations
    thread_combinations = [tuple([100] * n_combination)]

    # Iterate through all possible combinations of x, y, and z
    # Generate all possible combinations of thread percentages for n_combination workloads
    for combination in itertools.product(possible_values, repeat=n_combination):
        if sum(combination) == 100:
            thread_combinations.append(combination)

    
    header = []
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(n_combination):
            header += [f'workload{i+1}']
        #header += [f'(w1_{x}, w2_{y}, w3_{z})' for x,y,z in thread_combinations]
        # Generalize the header creation for thread combinations based on n_combination
        comb_labels = [f"({', '.join([f'w{i+1}_{thread}' for i, thread in enumerate(threads)])})"
                       for threads in thread_combinations]
        
        header += comb_labels
        csvwriter.writerow(header)

        for workloads, throughput_dict in share_file_steps.items():
            row = list(workloads)
            for threads in thread_combinations:
                throughput_per_combination = []
                for idx in range(n_combination):
                    thread = threads[idx]
                    workload = workloads[idx]
                    key = f'w{idx+1}_{workload}_MPS{thread}'
                    value = throughput_dict.get(key, None)
                    throughput_per_combination.append(value)
                row.append(tuple(throughput_per_combination))
            csvwriter.writerow(row)


import csv
def save_share_file_steps(share_file_steps,filename):
    # Define the thread combinations
    thread_combinations = [(10, 90), (20, 80), (30, 70), (40, 60), (50, 50), (60, 40), (70, 30), (80, 20), (90, 10), (100, 100)]

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write header
       
        header = ['workload1', 'workload2'] + [f'(LS{ls}, BE{be})' for ls, be in thread_combinations]
        csvwriter.writerow(header)
        
        # Write data
        for (workload1, workload2), throughput_dict in share_file_steps.items():
            row = [workload1, workload2]
            for ls, be in thread_combinations:
                ls_key = f'LS{ls}_{workload1}'
                be_key = f'BE{be}_{workload2}'
                ls_value = throughput_dict.get(ls_key, None)
                be_value = throughput_dict.get(be_key, None)
                row.append((ls_value, be_value))
            csvwriter.writerow(row)

# Function to save the collected data as CSV
def get_all_base_results(directory, BEtype):
    
    #THREAD_PERCENTAGES = [i for i in range(10, 101, 10)]
    THREAD_PERCENTAGES = [100]
    BASELINE_THREADS = ["MPS" + str(i) for i in THREAD_PERCENTAGES]
    results = defaultdict(dict)

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith('BE'):
                BE_dir = os.path.basename(root).replace("BE_", "")
                batch_size = int(BE_dir.split("_")[-1][5:])
                file_path = os.path.abspath(os.path.join(root, file))
                print(f"Processing file: {file_path}, Batch size: {batch_size}")
                model_type = BE_dir+"-"+BEtype

                for MPSpercent in BASELINE_THREADS:
                    if MPSpercent + ".log" in file.split("_"):
                        if BEtype == "train" and parse_train_log_avgStep_file(file_path) is not None:
                            throughput = parse_train_log_avgStep_file(file_path) * batch_size
                            results[model_type][f'Exclusive{MPSpercent[3:]}'] = throughput
                        else:
                            if parse_inf_log_avgStep_file(file_path) is not None:
                                throughput = parse_inf_log_avgStep_file(file_path) * batch_size
                                results[model_type][f'Exclusive{MPSpercent[3:]}'] = throughput
                        break
                        
                        

                            

                    # If gpu_info.csv file, calculate sm% and mem%
                    if MPSpercent in file and file.endswith("_gpu_info.csv"):
                        filepath = os.path.join(root, file)
                        avg_sm, avg_mem = calculate_average_sm_memory(filepath)
                        results[model_type]['sm%'] = avg_sm
                        results[model_type]['mem%'] = avg_mem

                    # If gpu_mem.csv file, calculate memory cap (GB)
                    if MPSpercent in file and file.endswith("_gpu_mem.csv"):
                        filepath = os.path.join(root, file)
                        gb_max = parse_memory_used_to_gb(filepath)
                        results[model_type]['memcap'] = gb_max
   

    # Save the results to CSV
    print(results)
    return results
    #save results in csv, with results first key as rows, second ket as columns


def save_baseline_to_csv(results, output_file):
    # Define the header
    header = ['Type', 'Exclusive100', 'sm%', 'mem%', 'memcap', 'Exclusive50']

    # Open the output CSV file in write mode
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the header
        writer.writerow(header)

        # Write each row from result_dict
        for model_type, metrics in results.items():
            writer.writerow([
                model_type,                # Type (model name)
                metrics.get('Exclusive100', ''),  # Exclusive100
                metrics.get('sm%', ''),           # sm%
                metrics.get('mem%', ''),          # mem%
                metrics.get('memcap', ''),         # memcap
                metrics.get('Exclusive50', '')         # memcap
            ])



if __name__ == "__main__":
    # Provide the filename as an argument
    filename = sys.argv[1]
    #parse_inf_log_avgStep_file("/Users/bing/Downloads/BE_openai/whisper-large-v2_batch2/BE_speech-recognition_MPS100.log")

    #get_multiworkloads_share_avgStep(filename,3, )

    # Calculate average sm and memory
    #avg_sm, avg_mem = calculate_average_sm_memory(filename)
    # Example usage
    csv_file = '/home/cc/mlProfiler/tests/mps/ccv100/baseline/batch4_mem/train/RUN1/LS0/speech-recognition/openai/whisper-large-v2_batch1/BE_bert-base-cased_batch4/BE_recommend_MPS100_gpu_mem.csv'  # Replace with your actual file path
    gb_values = parse_memory_used_to_gb(csv_file)
    print(gb_values)
    output_file = f"baseline_metrics.csv"
    # Save the average sm and memory to a new CSV file
    dirname = sys.argv[1]
    
    BEtypes = ["train", "inf"]  # or "train"
    result_dict = defaultdict(dict)
    for BEtype in BEtypes:
        
    #save_to_csv(avg_sm, avg_mem, filename)
        result = get_all_base_results(f"{dirname}/{BEtype}", BEtype)
        #update result_dict with result
        result_dict.update(result)
    save_baseline_to_csv(result_dict, f"baseline_metrics.csv")
    #save result_dict to csv

        
