import os
import matplotlib.pyplot as plt
import sys
import numpy as np
##### GLOBAL VARIABLES edit in here#####
rule_models = ["Oracle", "Random", "sm%", "mem%", "memcap", "Best-rule-based", '_Compute (SM) Throughput', "exclusive_throughput"]

filename_type_dict = {'wav2vec2-base-960h' : 'Wav2vec2', 'bert-base-cased': 'BERT',
                      'mobilenet': 'mobile', 'mobilenet_v2_1.0_224' : 'mobile','vit' : 'ViT', 
                      'vit_h_14' : 'ViT', 'whisper-large-v2': 'Whisper', 'vit-base-patch16-224' : "ViT",
                      'albert-base-v2' : 'ALBERT'}

key_color_dict = {'_Threads': 'r', '_Compute (SM) Throughput': 'c', '_DRAM Throughput': 'm', '_Memory Throughput': 'y', 
                  '_Registers': 'pink', 'exclusive_throughput': 'maroon', 'sm%': 'y', 'mem%': 'green', 'memcap': '#82cbec', 
                  'ground truth': 'b', "Oracle": "b", 'ground_truth value': 'b',
                   'Random': 'orange', 'random': 'orange', 'prediction': 'black', 'KACE': 'black','KACE_LR': 'black', 'LR': 'black' ,
                  'hotcloud_RF': 'r', 'hotcloud': 'r', 'Xu et al.':'r', 'Best-rule-based': 'gray', 'best_rule_based_value': 'gray', "NN": "c", "RF": "m", "AutoML": "r",
                 }
label_model_dict = {'Oracle': 'Oracle', 'ground truth': 'Oracle',"LR": "KACE", 'KACE': 'KACE_LR', 'hotcloud': 'Xu et al.','Best-rule-based': "Best rule", 'best_rule_based_value': 'Best rule based', 'sm%': 'SM%', 'mem%': 'MEM%', 'memcap': 'MEMCAP', 'random': 'Random',  'Random': 'Random', 'NN': 'KACE_NN', 'RF': 'KACE_RF', 'AutoML': 'KACE_AutoML', 
                    "_Compute (SM) Throughput": "Compute throughput", 'exclusive_throughput':'exclusive_throughput'}
#########################################

def parse_mse_r2_file(file_path):
    mse, r2 = None, None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "MSE:" in line:
                mse = float(line.split(":")[1].strip())
            if "R2:" in line:
                r2 = float(line.split(":")[1].strip())
    return mse, r2

def parse_predicted_file(file_path):
    predicted_throughput, max_throughput = None, None
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "predicted throughput:" in line:
                predicted_throughput = float(line.split(":")[1].strip())
            if "max throughput:" in line:
                max_throughput = float(line.split(":")[1].strip())
    if predicted_throughput is not None and max_throughput is not None:
        normalized_throughput = predicted_throughput / max_throughput
        return normalized_throughput
    return None

from collections import defaultdict
# Function to collect MSE, R2 and normalized throughput data
def collect_data(base_dir, n_runs):
    data = defaultdict(lambda: defaultdict(list))
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'mse_r2.txt'and "predictthroughput" in root:
                train_ratio = root.split('/')[-4]
                model = root.split('/')[-2]
                if model == 'KACE':
                    model = "LR"
                print(f"train ratio: {train_ratio}, model: {model}")
                print(f"mse r2 file read: {os.path.join(root, file)}")
                mse, r2 = parse_mse_r2_file(os.path.join(root, file))
                if mse is not None and r2 is not None:
                    data[train_ratio][model].append(('mse', mse))
                    data[train_ratio][model].append(('r2', r2))
            if file == 'predicted.txt' and "predictthroughput" in root:
                train_ratio = root.split('/')[-4]
                model = root.split('/')[-2]
                if model == 'KACE':
                    model = "LR"
                normalized_throughput = parse_predicted_file(os.path.join(root, file))
                if normalized_throughput is not None:
                    data[train_ratio][model].append(('throughput', normalized_throughput))
    print(data['trainratio_0.7']['AutoML'])
    avg_data = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for ratio, models in data.items():
        for model, metrics in models.items():
            for metric_type, value in metrics:
                avg_data[ratio][model][metric_type] += value
            avg_data[ratio][model] = {k: v / (len(metrics) / n_runs) for k, v in avg_data[ratio][model].items()}
    
    return avg_data


def collect_mse_r2_data(base_dir):
    train_ratios = {}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'mse_r2.txt':
                train_ratio = root.split('/')[-4]
                if train_ratio not in train_ratios:
                    train_ratios[train_ratio] = {'mse': [], 'r2': []}
                
                mse, r2 = parse_mse_r2_file(os.path.join(root, file))
                if mse is not None and r2 is not None:
                    train_ratios[train_ratio]['mse'].append(mse)
                    train_ratios[train_ratio]['r2'].append(r2)
    
    avg_mse_r2 = {}
    print(train_ratios)
    for ratio, values in train_ratios.items():
        avg_mse_r2[ratio] = {
            'mse': sum(values['mse']) / len(values['mse']) if values['mse'] else None,
            'r2': sum(values['r2']) / len(values['r2']) if values['r2'] else None
        }
    #save avg_mse_r2 to a file
    with open(f'{base_dir}/avg_mse_r2.txt', 'w') as f:
        f.write("Train Ratio\tMSE\tR2\n")
        for ratio, values in avg_mse_r2.items():
            f.write(f"{ratio}\t{values['mse']}\t{values['r2']}\n")        
    return avg_mse_r2

def collect_normalized_throughput(base_dir):
    train_ratios = {}
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'predicted.txt':
                train_ratio = root.split('/')[-4]
                if train_ratio not in train_ratios:
                    train_ratios[train_ratio] = []
                
                normalized_throughput = parse_predicted_file(os.path.join(root, file))
                if normalized_throughput is not None:
                    train_ratios[train_ratio].append(normalized_throughput)
    
    avg_normalized_throughput = {}
    print(train_ratios)
    for ratio, values in train_ratios.items():
        avg_normalized_throughput[ratio] = sum(values) / len(values) if values else None
    
    # Save avg_normalized_throughput to a file
    with open(f'{base_dir}/avg_normalized_throughput.txt', 'w') as f:
        f.write("Train Ratio\tNormalized Throughput\n")
        for ratio, value in avg_normalized_throughput.items():
            f.write(f"{ratio}\t{value}\n")
    return avg_normalized_throughput


def plot_data(avg_data, output_dir):
    train_ratios = sorted(avg_data.keys())
    models = ['RF', 'NN', 'AutoML', 'LR']
    legend_order = ['KACE_RF', 'KACE_NN', 'KACE_AutoML', 'KACE_LR']
    colors = {'KACE_RF': 'purple', 'KACE_NN': 'c', 'KACE_AutoML': 'r', 'KACE_LR': 'black'}
    

    label_font = 19 
    axis_font = 19
    legend_font = 12
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    for model in models:
        mse_values = [avg_data[ratio][model]['mse'] for ratio in train_ratios]
        r2_values = [avg_data[ratio][model]['r2'] for ratio in train_ratios]
        throughput_values = [avg_data[ratio][model]['throughput'] for ratio in train_ratios]
        
        train_ratios_float = [float(r.split('_')[1]) for r in train_ratios]
        train_ratios_percent = [100 * i for i in train_ratios_float]
                
        #axs[0].plot(train_ratios_percent, r2_values, label=f'KACE_{model}', color=colors[f'KACE_{model}'], marker='o')
        #axs[1].plot(train_ratios_percent, mse_values, label=f'KACE_{model}', color=colors[f'KACE_{model}'], marker='o')
        #axs[2].plot(train_ratios_percent, throughput_values, label=f'KACE_{model}', color=colors[f'KACE_{model}'], marker='o')
        axs[0].plot(train_ratios_percent, mse_values, label=f'KACE_{model}', color=colors[f'KACE_{model}'], marker='o')
        axs[1].plot(train_ratios_percent, throughput_values, label=f'KACE_{model}', color=colors[f'KACE_{model}'], marker='o')

    
    """
    axs[0].set_ylabel('R²', fontsize=axis_font)
    axs[0].set_ylim(0, 1.1)
    #axs[1].legend(legend_order)
    axs[0].grid(True)
    axs[0].set_xlabel('Training set size (%)', fontsize= 12)
    """

    axs[0].set_ylabel('MSE', fontsize=axis_font)
    axs[0].set_ylim(0, max([max(avg_data[ratio][model]['mse'] for ratio in train_ratios) for model in models]) * 1.1)
    #axs[1].legend(legend_order, ncol=4, fontsize=legend_font)
    axs[0].grid(True)
    axs[0].set_xlabel('Training set size (%)', fontsize= axis_font)

    #axs[1].set_xlabel('Training set size (%)', fontsize= 12)
    

    axs[1].set_xlabel('Training set size (%)', fontsize=axis_font)
    axs[1].set_ylabel('Normalized throughput sum',fontsize=axis_font-2)
    axs[1].set_ylim(0, 1.1)

    axs[1].grid(True)
    for i in range(len(axs)):
        for tick in axs[i].xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font)
        for tick in axs[i].yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font)
 

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    #plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    #plt.show()
def parse_rulebase_file(file_path, rule_models, data):

    with open(file_path, 'r') as f:
        for line in f:
            for model in rule_models:
                if line.startswith(model):
                     if model == "Oracle":
                         max_throughput = float(line.split(":")[1].strip())
                         data[model]['throughput'].append(float(1.0))
                         
                     else:
                        data[model]['throughput'].append(float(line.split(":")[1].strip())/ max_throughput)
    #add 
    #raisevalueerror if rule_models is not in data
    for model in rule_models:
        if model != "Best-rule-based" and model not in data:
            raise ValueError(f"rule {model} not found in data")
    #add best rule based
    data["Best-rule-based"]['throughput'].append(max(data[model]['throughput'][-1] for model in ["sm%", "mem%", "memcap"]))
    return data

def collect_unseen_data(base_dir):
    data = defaultdict(lambda: defaultdict(list))
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'mse_r2.txt':
                #print(root)
                model = root.split('/')[-3]
                if model == 'KACE':
                    model = "LR"
                #train_ratio = root.split('/')[-4]
                
                mse, r2 = parse_mse_r2_file(os.path.join(root, file))
                if mse is not None and r2 is not None:
                    data[model]['mse'].append(mse)
                    data[model]['r2'].append(r2)
            elif file == 'predicted.txt':
                model = root.split('/')[-3]
                if model == 'KACE':
                    model = "LR"
                train_ratio = root.split('/')[-4]
                
                normalized_throughput = parse_predicted_file(os.path.join(root, file))
                if normalized_throughput is not None:
                    #print( model ,normalized_throughput)
                    data[model]['throughput'].append(normalized_throughput)
            elif file == 'rulebase.txt' and  root.split('/')[-3] == "KACE":
                #only parse rulebase file for KACE
                #parse Random, sm%, mem%, memcap, best-rule-based here
                data = parse_rulebase_file(os.path.join(root, file), rule_models, data)

                
                
    #print(data)
    avg_data = {}
   
    for model, metrics in data.items():
        print(f"model: {model}, metrics: {metrics}")
        avg_data[model] = {
            'mse': sum(metrics['mse']) / len(metrics['mse']) if metrics['mse'] else None,
            'r2': sum(metrics['r2']) / len(metrics['r2']) if metrics['r2'] else None,
            'throughput': sum(metrics['throughput']) / len(metrics['throughput']) if metrics['throughput'] else None
        }
    print(avg_data)
    return avg_data

### Step 2: Define the Function to Plot the Data


def plot_unseen_data(avg_data, output_dir, baselines):
    """
    models = ['RF', 'NN', 'AutoML', 'LR']
    
    legend_order = ['KACE_RF', 'KACE_NN', 'KACE_AutoML', 'KACE_LR']
    colors = {'KACE_RF': 'purple', 'KACE_NN': 'c', 'KACE_AutoML': 'r', 'KACE_LR': 'black'}
    Best-rule
    """
    label_font = 14 
    axis_font = 13
    legend_font = 12

    # Create individual figures
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    
    # Data for plotting
    
    throughput_values = [avg_data[model]['throughput'] for model in baselines]

    x_pos = np.arange(len(baselines))
    """
    mse_values = [avg_data[model]['mse'] for model in models]
    r2_values = [avg_data[model]['r2'] for model in models]
    # Plot R²
    bars = axs[0].bar(x_pos, r2_values, color=[colors[f'KACE_{model}'] for model in models])
    axs[0].set_ylabel('R²', fontsize=axis_font)
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(legend_order, rotation=0, ha='center', fontsize=axis_font)
    axs[0].set_ylim(0, 1.1)
    axs[0].grid(True)
    # Add text boxes for R²
    for bar in bars:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='0.85', alpha=0.8))
    
    # Plot MSE
    bars = axs[1].bar(x_pos, mse_values, color=[colors[f'KACE_{model}'] for model in models])
    axs[1].set_ylabel('MSE', fontsize=axis_font)
    axs[1].set_xticks(x_pos)
    #axs[1].set_xticklabels(legend_order, rotation=45, ha='right', fontsize=axis_font)
    axs[1].set_xticklabels(legend_order, rotation=0, ha='center', fontsize=axis_font)
    axs[1].set_ylim(0, max(mse_values) * 1.1)
    axs[1].grid(True)
    for bar in bars:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='0.85', alpha=0.8))
    """
    # Plot Normalized Throughput
    bars = axs.bar(x_pos, throughput_values, color=[key_color_dict[f'{model}'] for model in baselines])
    axs.set_ylabel('Normalized throughput sum', fontsize=axis_font-2)
    axs.set_xticks(x_pos)
    axs.set_xticklabels([label_model_dict[f'{model}'] for model in baselines], rotation=0, ha='center', fontsize=axis_font)
    axs.set_ylim(0, 1.1)
    axs.grid(True)
    #axs.set_xlabel('Models', fontsize=label_font)
    for bar in bars:
        height = bar.get_height()
        axs.text(bar.get_x() + bar.get_width() / 2.0, 0.2, f'Gain\n{height*100:.1f}%', ha='center', va='bottom', fontsize=12, bbox=dict(facecolor='0.85', alpha=0.7))

    # Adjust ticks and labels
    for i in range(3):
        for tick in axs.xaxis.get_major_ticks():
            tick.label.set_fontsize(12)
        for tick in axs.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        #axs[i].set_xlabel('Models', fontsize=label_font)
    #set title
    axs.set_title('gpt2-xl unseen', fontsize=label_font+2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'unseen_metrics_plot.png'))




"""
# Base directory where your output is located
base_dir = sys.argv[1]
# Collect MSE and R2 data
avg_mse_r2 = collect_mse_r2_data(base_dir)
# Plot the data
plot_mse_r2(avg_mse_r2, base_dir)
# Collect normalized throughput data
avg_normalized_throughput = collect_normalized_throughput(base_dir)
# Plot the data
plot_normalized_throughput(avg_normalized_throughput, base_dir)
"""

# Base directory where your output is located
#python plot_mse_r2.py /Users/bing/Documents/mlProfiler/tests/mps/multiinstance/output/run0730_KACE_batch2-8/
#error check args
if len(sys.argv) < 2:
    print("please provide the output file name")
    print("python plot_mse_r2.py output_file_name")
    exit(1)

base_dir = sys.argv[1]

# Collect data
#avg_data = collect_data(base_dir, n_runs=3)

# Plot the data
#plot_data(avg_data, "/Users/bing/Documents/RPE/ch7_conclusion")

# Collect unseen data
avg_unseen_data = collect_unseen_data(f"{base_dir}/unseen_partition")

# Plot the data
#['RF', 'NN', 'AutoML', 'LR']
#plot_unseen_data(avg_unseen_data, "/Users/bing/Documents/RPE/ch7_conclusion", baselines=['RF', 'NN', 'AutoML', 'LR'])
#[ 'Oracle','LR', "sm%", "mem%", "memcap", "Best-rule-based"]
plot_unseen_data(avg_unseen_data, f"{base_dir}", baselines=[ 'Oracle','LR', "hotcloud","Best-rule-based", "memcap", "sm%", "mem%","Random" ])
#plot_unseen_data(avg_unseen_data, f"{base_dir}", baselines=[ 'Oracle','LR',"Best-rule-based", "memcap", "sm%", "mem%","Random"])