import json
import sys
import pandas as pd
def calculate_averages(filename):
    cpu_percents = []
    mem_percents = []

    with open(filename, 'r') as file:
        for line in file:
            try:
                
                    #strip all spaces
                key, value = line.strip().split(":",1)
                value = value.strip().replace("\"", "").replace("%", "").replace(",", "").strip()
                
                #print(f"Key: {key}, Value: {value}, Type: {type(value)}, Length: {len(value)}")

                #get the string after  :
                #key, value = line.split(":")
                #print(key, value)  
                #print("CPUPerc" in key.strip())
                key = key.strip("\"")
                #print("key", key.strip("\""), f"length of key={len(key)}")
                if  "CPUPerc" in key.strip("\""):

                    cpu_percents.append(float(value))
                    
                elif "MemPerc" in key.strip() :
                    mem_percents.append(float(value.strip("%")))
            except ValueError:
                print(f"not parsed: {line}")
                continue
    print(cpu_percents, mem_percents)
    avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
    avg_mem = sum(mem_percents) / len(mem_percents) if mem_percents else 0
    #save average CPU and Memory to csv


    return avg_cpu, avg_mem
def split_train_test(originTrainfile, labelfilename):
    traindf= pd.read_csv(originTrainfile)
    labeldf = pd.read_csv(labelfilename)
    #find workload1 , workload2 pairs in traindf and use it to match label df
    #get all unique workload1, workload2 in traindf
    train_idx = []
    for i ,row in traindf.iterrows():
        
        workload1 = row['workload1']
        workload2 = row['workload2']
        #find the corresponding idx in labeldf
        print("current train pair", workload1, workload2)
        
        label = labeldf[(labeldf['workload1'] == workload1) & (labeldf['workload2'] == workload2)]
        if label.empty:
            label = labeldf[(labeldf['workload1'] == workload2) & (labeldf['workload2'] == workload1)]
        #add index of label in labeldf to train_idx
        if label.empty:
            continue
        print(label)
        train_idx.append(label.index[0])
        
    #split labeldf into train and test using train_idx
    testdf = labeldf.drop(train_idx)
    traindf = labeldf.loc[train_idx]
    testdf.to_csv('testing_set_MPS100.csv', index=False)
    traindf.to_csv('training_set_MPS100.csv', index=False)
    


    
if __name__ == "__main__":
    #filename = sys.argv[1]  # Update with your file path
    #avg_cpu, avg_mem = calculate_averages(filename)
    
    #print(f"Average CPU %: {avg_cpu:.2f}%")
    #print(f"Average Memory %: {avg_mem:.2f}%")
    split_train_test(sys.argv[1], sys.argv[2])
    #example - split_train_test('/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/MPS100/training_set.csv', 'mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_labels.csv_targetMPS100.csv')
