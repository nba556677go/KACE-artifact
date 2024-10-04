import nbformat
from nbconvert import PythonExporter
import importlib.util
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict

import logging

# Configure the logger
logging.basicConfig(
    format='%(levelname)s[%(filename)s - %(funcName)s - line:%(lineno)d ]  - %(message)s',
    level=logging.INFO
)


def getcloud_stage1trainData(baselineData,shareThroughputData, kernelData,outname,targetMPS, selected_feats):
    base_df = pd.read_csv(baselineData)
    #drop row with nan values
    base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
    base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
    #base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
    base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
    base_CPU_dict = base_df.set_index('Type')['AvgCPU'].to_dict()
    base_MainMem_dict = base_df.set_index('Type')['AvgMem'].to_dict()


    #print(base_sm_dict)

    # Step 1: Read the first CSV file
    shared_throughput = pd.read_csv(shareThroughputData )
    if targetMPS != 100 :
        targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    #only keep columns with selected feats
    #add w1_ and w2_ prefix to all columns in kernel_file
    selected_feats = ['w1_'+ col for col in selected_feats] + ['w2_'+ col for col in selected_feats]
    #add wrorkload1 and workload2 to selected_feats
    selected_feats += ['workload1', 'workload2', 'idx1', 'idx2'] 

    kernel_file = kernel_file[selected_feats]

    
    columns = shared_throughput.columns
    #get column pos if  targetMPS in column
    targetMPS_pos = [i for i, col in enumerate(columns) if f'LS{targetLS}, BE{targetBE}' in col][0]
    #print(targetMPS_pos)
    target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list()]
    #get the entrue row of targetMPS_pos
    shared_throughput[[f'LS{targetLS}', f'BE{targetBE}']] = pd.DataFrame(target_throughput, index=shared_throughput.index)
    #print(shared_throughput[[f'LS{targetMPS}', f'BE{100-targetMPS}']])
    # Step 3: Extract the (LS100, BE100) pairs and create a DataFrame
    
    #shared_throughput = shared_throughput.drop(columns=[f'LS{targetMPS}, BE{100-targetMPS}'])

    # Step 4: Create a mapping for both regular and reversed (workload1, workload2) pairs
    mapping = {}
    for _, row in shared_throughput.iterrows():
        w1, w2 = row['workload1'], row['workload2']
        ls, be = row[f'LS{targetLS}'], row[ f'BE{targetBE}']
        mapping[(w1, w2)] = (ls, be)
        mapping[(w2, w1)] = (be, ls)

    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    for _, row in kernel_file.iterrows():
        workload1 = row['workload1']
        workload2 = row['workload2']
        ls100_be100 = mapping.get((workload1, workload2), (None, None))
        if ls100_be100[0] is None or ls100_be100[1] is None:
            no_data.append((workload1, workload2))
            #continue  # Skip if any element of the tuple is None
        merged_row = row.tolist() + [ls100_be100[0], ls100_be100[1]]
        merged_rows.append(merged_row)

    # Step 6: Create a DataFrame from the merged rows
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + ['w1throughput', 'w2throughput'])
    #drop nan values
    merged_df.dropna(inplace=True)

    print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    merged_df ['w1exclusive_throughput'] = merged_df['workload1'].map(base_throughput_dict)
    merged_df ['w2exclusive_throughput'] = merged_df['workload2'].map(base_throughput_dict)

    #apply sm and mem% to csv_data using base_sm_dict and base_mem_dict
    merged_df ['w1sm%'] = merged_df ['workload1'].map(base_sm_dict)
    merged_df ['w2sm%'] = merged_df ['workload2'].map(base_sm_dict)
    merged_df ['w1mem%'] = merged_df ['workload1'].map(base_mem_dict)
    merged_df ['w2mem%'] = merged_df ['workload2'].map(base_mem_dict)
    #merged_df ['w1memcap'] = merged_df ['workload1'].map(base_memcap_dict)
    #merged_df ['w2memcap'] = merged_df ['workload2'].map(base_memcap_dict)
    merged_df ['w1CPU%'] = merged_df ['workload1'].map(base_CPU_dict)
    merged_df ['w2CPU%'] = merged_df ['workload2'].map(base_CPU_dict)
    merged_df ['w1MainMem%'] = merged_df ['workload1'].map(base_MainMem_dict)
    merged_df ['w2MainMem%'] = merged_df ['workload2'].map(base_MainMem_dict)
    merged_df.dropna(inplace=True)

    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname
    





def train_test_split_with_counts(data,  train_outname, test_outname, n_workload_counts, random_seed):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(data)
    # Create a dictionary to count the appearances of each workload
    #shuffle df
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    workload_count = defaultdict(int)

    # Create a list to store the indices of rows to include in the training set
    training_indices = []

    # Iterate over each row in the DataFrame
    #ensure that each workload appears at most n_workload_counts times
    while (all(value < n_workload_counts for value in workload_count.values())):
        for idx, row in df.iterrows():
            workload1 = row['workload1']
            workload2 = row['workload2']
            if workload1 == workload2:
                continue
            # Check if adding this row will exceed the count of 2 for either workload
            if workload_count[workload1] < n_workload_counts or workload_count[workload2] < n_workload_counts:
                training_indices.append(idx)
                workload_count[workload1] += 1
                workload_count[workload2] += 1

    # Create the training set DataFrame
    training_set = df.loc[training_indices]
    testing_set = df.drop(training_indices)
    #get train_outname absoulte path
    #train_outname = os.path.abspath(train_outname)
    print(f"training set save to {os.path.abspath(train_outname)}")
    print(f"testing set save to {os.path.abspath(test_outname)}")
    # Save the training set to a new CSV file
    training_set.to_csv(train_outname, index=False)
    #create testing_set with the rest of the data

    # Save the testing set to a new CSV file
    testing_set.to_csv(test_outname, index=False)


    return  train_outname, test_outname


def getstage1trainData(baselineData, shareThroughputData , kernelData, outname, targetMPS):
    #combine baseline kernel and shared throughput as training target
    
    base_df = pd.read_csv(baselineData)
    #drop row with nan values
    base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
    base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
    base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
    base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()

    #print(base_sm_dict)

    # Step 1: Read the first CSV file
    shared_throughput = pd.read_csv(shareThroughputData )
    if targetMPS != 100 :
        targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    columns = shared_throughput.columns
    #get column pos if  targetMPS in column
    targetMPS_pos = [i for i, col in enumerate(columns) if f'LS{targetLS}, BE{targetBE}' in col][0]
    #print(targetMPS_pos)
    target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list()]
    #get the entrue row of targetMPS_pos
    shared_throughput[[f'LS{targetLS}', f'BE{targetBE}']] = pd.DataFrame(target_throughput, index=shared_throughput.index)
    #print(shared_throughput[[f'LS{targetMPS}', f'BE{100-targetMPS}']])
    # Step 3: Extract the (LS100, BE100) pairs and create a DataFrame
    
    #shared_throughput = shared_throughput.drop(columns=[f'LS{targetMPS}, BE{100-targetMPS}'])

    # Step 4: Create a mapping for both regular and reversed (workload1, workload2) pairs
    mapping = {}
    for _, row in shared_throughput.iterrows():
        w1, w2 = row['workload1'], row['workload2']
        ls, be = row[f'LS{targetLS}'], row[ f'BE{targetBE}']
        mapping[(w1, w2)] = (ls, be)
        mapping[(w2, w1)] = (be, ls)

    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    for _, row in kernel_file.iterrows():
        workload1 = row['workload1']
        workload2 = row['workload2']
        ls100_be100 = mapping.get((workload1, workload2), (None, None))
        if ls100_be100[0] is None or ls100_be100[1] is None:
            no_data.append((workload1, workload2))
            #continue  # Skip if any element of the tuple is None
        merged_row = row.tolist() + [ls100_be100[0], ls100_be100[1]]
        merged_rows.append(merged_row)

    # Step 6: Create a DataFrame from the merged rows
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + ['w1throughput', 'w2throughput'])
    #drop nan values
    merged_df.dropna(inplace=True)

    print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    merged_df ['w1exclusive_throughput'] = merged_df['workload1'].map(base_throughput_dict)
    merged_df ['w2exclusive_throughput'] = merged_df['workload2'].map(base_throughput_dict)

    #apply sm and mem% to csv_data using base_sm_dict and base_mem_dict
    merged_df ['w1sm%'] = merged_df ['workload1'].map(base_sm_dict)
    merged_df ['w2sm%'] = merged_df ['workload2'].map(base_sm_dict)
    merged_df ['w1mem%'] = merged_df ['workload1'].map(base_mem_dict)
    merged_df ['w2mem%'] = merged_df ['workload2'].map(base_mem_dict)
    merged_df ['w1memcap'] = merged_df ['workload1'].map(base_memcap_dict)
    merged_df ['w2memcap'] = merged_df ['workload2'].map(base_memcap_dict)
    merged_df.dropna(inplace=True)

    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname
    
def filter_data(data, workload,  custom_col_exclude, n_combination ,isbatchThroughput = False):
                         

    print("FILTERED workload: ", workload)
    # Filter data by workload name == workload1 or workload2
    #remove all workload that contains mobilenet_v2_1.0_224 and resnet-50 and mobilenet_
    #data = data[~data['workload1'].str.contains('mobilenet_v2_1.0_224') & ~data['workload2'].str.contains('mobilenet_v2_1.0_224')]
    #data = data[~data['workload1'].str.contains('resnet-50') & ~data['workload2'].str.contains('resnet-50')]
    #data = data[~data['workload1'].str.contains('mobilenet_') & ~data['workload2'].str.contains('mobilenet_')]

    #filter out data with batch size 8
    for i in range(1, n_combination+1):
        data[f'w{i}batch_size'] = data[f'workload{i}'].str.extract(r'_batch(\d+)', expand=False).astype(int)
        for col in custom_col_exclude:
            data = data.drop(columns=[f'w{i}{col}'])
    #data['w1batch_size'] = data['workload1'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    #data['w2batch_size'] = data['workload2'].str.extract(r'_batch(\d+)', expand=False).astype(int)
    #data = data[(data['w1batch_size'] == 8) & (data['w2batch_size'] == 8)]

    #exclude data with both trains
    #data = data[~data['workload1'].str.contains('train') | ~data['workload2'].str.contains('train')]
    #data = data[~data['workload1'].str.contains('wav2vec2-base-960h') & ~data['workload2'].str.contains('wav2vec2-base-960h')]
    #filter data = data with workload1 == workload or workload2 == workload
    
    # Divide throughput and exclusive throughput by batch size
    if isbatchThroughput:
        for i in range(1, n_combination+1):
            data[f'w{i}throughput'] = data[f'w{i}throughput'] / data[f'w{i}batch_size']
            data[f'w{i}exclusive_throughput'] = data[f'w{i}exclusive_throughput'] / data[f'w{i}batch_size']

    #print(f"cols custom to be excluded{custom_col_exclude}")
    #print(f"prefilter data columns{data.columns}")
    #if custom_col_exclude == []:
    #    raise ValueError("should include something")
    if workload != "":
        condition = pd.Series([False] * len(df))
        for i in range(1, n_combination+1):
            condition = condition | df[f'workload{i}'].str.contains(workload)
        data = data[condition]

    for i in range(1, n_combination+1):
        data = data.drop(columns=[f'w{i}batch_size'])
    if workload == "":
        #drop workload1 == mobile-inf_batch64  workload2 == mobile-inf_batch64
        return data
    #concat data with workload1 and workload2 that contains workload
    #print(data[data['workload1'].str.contains(workload)])
    #data = data[(data['workload1'].str.contains(workload) ) | (data['workload2'].str.contains(workload) )]
    
    #df = df1.append(data[data['workload2'].str.contains(workload)])
    #drop batch size columns
    
    

    return data

def preprocess_data(data, categorical_columns, target, correlation, n_combination):
        # Drop non-numerical and non-relevant columns
        #drop categorical columns
        logging.info(f"preprocess columns: {data.columns.tolist()}")
        print(f"n_combination: {n_combination}")
        data[target] = 0 
        for i in range(1, n_combination+1):
            if target == 'L2norm':
                
                data[target] += np.sqrt((data[f'w{i}throughput'] / data[f'w{i}exclusive_throughput'])**2)
                #data[target] = np.sqrt((data['w1throughput'] / data['w1exclusive_throughput'])**2 + (data['w2throughput'] / data['w2exclusive_throughput'])**2)
            elif target == 'sum_throughput':
                data[target] += data[f'w{i}throughput']
                #logging.info(data[f'w{i}throughput'])
                #data[target] = data['w1throughput'] + data['w2throughput'] 
            else:
                logging.error("target not found")
                exit(1)
        #logging.info(f"target: {data[target]}")
        #data = data.drop(columns=categorical_columns)
        #get a set of feature colunmn names after removing w1 prefix
        feat_names = [col[2:] for col in data.columns if 'w1' in col] 
        logging.info(f"all feature names should be used with target : {feat_names}")
        #remove throughput  columen since they were used to calculate L2norm
        columns_excluded = categorical_columns 
        for i in range(1, n_combination+1):
            columns_excluded +=  [f'w{i}throughput']
        
        for feat in feat_names:
            #add a new column with the name feat
            if f'w1{feat}' in  columns_excluded:
                print("excluded", f'w1{feat}')
                continue 
            data[feat] = 0
            for i in range(1, n_combination+1):
                #print(f"adding w{i}{feat} = \n{data[f'w{i}{feat}']}")
                data[feat] += data[f'w{i}{feat}']
                columns_excluded.append(f'w{i}{feat}')
            #print(f"aggreate {feat} = \n{data[feat]}")
            if '%' in feat:
                data[feat] = data[feat] / n_combination
            #data[feat] = (data[f'w1{feat}'] + data[f'w2{feat}'])/2  if '%' in feat else data[f'w1{feat}'] + data[f'w2{feat}']
                
            #columns_excluded.append(f'w1{feat}')
            #columns_excluded.append(f'w2{feat}')
       
        logging.info(f"columns excluded: {columns_excluded}")
        #columns_excluded = categorical_columns + ['w1throughput', 'w2throughput', 'w1exclusive_throughput', 'w2exclusive_throughput', 'w1sm%', 'w2sm%', 'w1mem%', 'w2mem%']
        #columns_excluded = categorical_columns 
        least_correlated_columns = get_least_corr_columns(data=data, target_column=target, excluded_columns=columns_excluded, CORRELATION=correlation)
        #data = data.drop(least_correlated_columns, axis=1)  # Drop least correlated columns
        drop_columns = columns_excluded + least_correlated_columns
        return data, drop_columns 
    


def get_least_corr_columns(data, target_column, excluded_columns, CORRELATION):
        
        
    #drop categorical columns or not using columns
    corr_data = data.drop(excluded_columns, axis=1)

    #z scale data
    corr_data = (corr_data - corr_data.mean()) / corr_data.std()  
    correlations = corr_data.corr()
    least_correlated_columns = list(correlations[target_column][abs(correlations[target_column]) < CORRELATION].index)
    logging.info(correlations[target_column].sort_values(ascending=False))
    logging.info(f"drop columns {least_correlated_columns}")
    logging.info(f"using columns {list(correlations[target_column][abs(correlations[target_column]) >= CORRELATION].index)}")
    return least_correlated_columns


def train_test_custom_split(data, test_workload, target, RANDOMSEED, CORRELATION, n_combination):
    test_workloads = pd.DataFrame()

    #get categorical columns
    categorical_columns = []
    for i in range(1, n_combination+1):
        categorical_columns += [f'workload{i}', f'idx{i}']
    
    data, columns_excluded = preprocess_data(data, categorical_columns, target, correlation=CORRELATION, n_combination=n_combination)
    print(f"columns excluded {columns_excluded}")
    #split data with wokload1 == berttrain of workload2 == berttrain

    if test_workload != "":

        condition = pd.Series([False] * len(data))
        for i in range(1, n_combination+1):
            condition = condition | data[f'workload{i}'].str.startswith(test_workload)
        test_set = data[condition]
        #test_set = data[(data['workload1'].str.startswith(test_workload) ) | (data['workload2'].str.startswith(test_workload) )]
        
        #set train_set to data excluding test_set
        train_set = data[~data.index.isin(test_set.index)]
    #print(train_set['workload1'], train_set['workload2'])
    #print(len(train_set))

        # Drop non-numerical and non-relevant columns
        train_set = train_set.drop(columns=columns_excluded)
        #add test_workloads
        for i in range(1, n_combination+1):
            test_workloads[f'workload{i}'] = test_set[f'workload{i}']

        test_set = test_set.drop(columns=columns_excluded)
        #normalize all data in train_set and test_set
        #do log normalization on train_set and test_set
        #train_set = np.log1p(train_set)
        #test_set = np.log1p(test_set)
        
        
        train_set = (train_set - train_set.mean()) / train_set.std()
        test_set = (test_set - test_set.mean()) / test_set.std()
        #set train and test
        X_train , X_test = train_set.drop(target, axis=1), test_set.drop(target, axis=1)
        y_train, y_test = train_set[target], test_set[target]
    else:
        #standard train test split
        from sklearn.model_selection import train_test_split
        #data = data[~data['workload1'].str.contains('mobilenet_v2_1.0_224') & ~data['workload2'].str.contains('mobilenet_v2_1.0_224')]
        #data = data[~data['workload1'].str.contains('resnet-50') & ~data['workload2'].str.contains('resnet-50')]
        #data = data[~data['workload1'].str.contains('mobilenet_') & ~data['workload2'].str.contains('mobilenet_')]
        
        X = data.drop(columns=[target])
        y = data[target]
        #X_train, X_test, y_train, y_test = train_test_split_with_counts(data, workload_counts=3, target=target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOMSEED)
        for i in range(1, n_combination+1):
            test_workloads[f'workload{i}'] = X_test[f'workload{i}']
        #test_workloads["workload1"] = X_test['workload1'] 
        #test_workloads["workload2"] = X_test['workload2']
        #drop non relevant columns
        X_train = X_train.drop(columns=columns_excluded)
        X_test = X_test.drop(columns=columns_excluded)
        #normalize all data in train_set and test_set
        X_train = (X_train - X_train.mean()) / X_train.std()
        X_test = (X_test - X_test.mean()) / X_test.std()
        y_train = (y_train - y_train.mean()) / y_train.std()
        y_test = (y_test - y_test.mean()) / y_test.std()
        print("Xtest", X_test)
        print("y_test", y_test)
        print(f"X_test columns: {X_test.columns}")
    
    return X_train, X_test, y_train, y_test, test_workloads, columns_excluded

def import_notebook(nb_path):
    # Load the notebook content
    with open(nb_path) as f:
        nb_content = nbformat.read(f, as_version=4)
    
    # Convert notebook to a Python script
    exporter = PythonExporter()
    source, _ = exporter.from_notebook_node(nb_content)
    
    # Save the converted script
    script_path = nb_path.replace('.ipynb', '.py')
    with open(script_path, 'w') as f:
        f.write(source)
    
    # Import the script as a module
    spec = importlib.util.spec_from_file_location("notebook_module", script_path)
    notebook_module = importlib.util.module_from_spec(spec)
    sys.modules["notebook_module"] = notebook_module
    spec.loader.exec_module(notebook_module)
    
    # Clean up the temporary Python script
    os.remove(script_path)
    
    return notebook_module

def get_multiinstance_stage1trainData(baselineData, shareThroughputData , kernelData, outname, targetMPS, n_combination, hotcloud=False):
    #combine baseline kernel and shared throughput as training target
    
    

    #print(base_sm_dict)

    # Step 1: create target MPS keys based on combinations
    
    if targetMPS != 100 :
        raise ValueError("targetMPS that != 100 is not yet implemented!")
        #targetLS, targetBE = targetMPS, 100-targetMPS 
    else:
        target_MPS_list = [100] * n_combination
        for i, mps_value in enumerate(target_MPS_list):
            target_MPS_list[i] = f"w{i+1}_{mps_value}"
        target_MPS_key = ", ".join(target_MPS_list)
        #targetLS, targetBE = 100,100
        
    # Step 2: Read the second CSV file
    kernel_file = pd.read_csv(kernelData)
    if hotcloud:
        #keep only kernels that we use columns
        selected_feats = ["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
        #add w1_ and w2_ prefix to all columns in kernel_file
        selected_feat_cols = []
        for i in range(1, n_combination+1):
            selected_feat_cols += [f'w{i}_'+ col for col in selected_feats]
            selected_feat_cols += [f'workload{i}']
            selected_feat_cols += [f'idx{i}']

        kernel_file = kernel_file[selected_feat_cols]

    #read shared throughput data
    shared_throughput = pd.read_csv(shareThroughputData)
    columns = shared_throughput.columns
    targetMPS_pos = [i for i, col in enumerate(columns) if target_MPS_key in col][0]
    #print(targetMPS_pos)
    #print(target_MPS_key)
   
   # target_throughput = defaultdict(tuple)
    target_throughput = [eval(y) for y in shared_throughput.iloc[:, targetMPS_pos].to_list()]
    #create a dict with (workloads) as keys, target_throughput as values
    #target_throughput = {row['workload1']: eval(row[target_MPS_key]) for _,row in shared_throughput.iterrows()}
    target_throughput_dict = defaultdict(list)
    for _, row in shared_throughput.iterrows():
        workload_instances = []
        throughput_values = eval(row.iloc[targetMPS_pos])  # Assuming `row.iloc[targetMPS_pos]` returns the list of throughput values
        
        # Collect all workload instances
        for i in range(n_combination):
            workload_instances.append(row[f'workload{i+1}'])
        
        # Zip workload instances with throughput values
        workload_with_values = list(zip(workload_instances, throughput_values))
        # Sort the zipped pairs based on workload instances
        workload_with_values_sorted = sorted(workload_with_values, key=lambda x: x[0])  # Sort by workload instance (the first item in each tuple)
        # Unzip the sorted pairs back into two lists: sorted workloads and sorted throughput values
        sorted_workloads, sorted_values = zip(*workload_with_values_sorted)
        # Insert the sorted key-value pair into the target_throughput_dict
        target_throughput_dict[tuple(sorted_workloads)] = sorted_values
    #get the entrue row of targetMPS_pos
    print(len(target_throughput_dict))
    # Step 5: Merge the DataFrames based on workload1 and workload2
    merged_rows = []
    no_data = []
    print(kernel_file.columns)
    for idx, row in kernel_file.iterrows():
        workload_instances = []
            
        # Collect all workload instances from the row
        for i in range(n_combination):
            workload_instances.append(row[f'workload{i+1}'])
        
        if idx == 299:
            print(workload_instances)
        # Sort the tuple to ensure the order doesn't matter for lookup
        sorted_workload_instances = tuple(sorted(workload_instances))
        if idx == 299:
            print(sorted_workload_instances)
        
        # None tuple is default if not getting mappings
        None_tuple = tuple([None] * n_combination)
        
        # Check the target throughput dictionary for the sorted tuple
        throughput_of_instances = target_throughput_dict.get(sorted_workload_instances, None_tuple)
        
        
        # If we found a throughput, rematch the order of throughput with original (unsorted) workload instances
        if None in throughput_of_instances:
            no_data.append(tuple(workload_instances))
            #print("no data in shared_throughput for the following pairs: ", tuple(workload_instances))

        else:
            # Map sorted workloads to the throughput values using list (not dict)
            
            sorted_pairs = list(zip(sorted_workload_instances, throughput_of_instances))
            # Create a list to hold the reordered throughput values
            reordered_throughput = []
            
            # Reorder the throughput according to the original order of workload_instances
            for workload in workload_instances:
                for sorted_workload, throughput in sorted_pairs:
                    if sorted_workload == workload:
                        reordered_throughput.append(throughput)
                        sorted_pairs.remove((sorted_workload, throughput))  # Avoid double-counting
                        break
            
            throughput_of_instances = tuple(reordered_throughput)
            #if "vit_h_14_batch2-train"in workload_instances[2]  and "albert-base-v2_batch2-train" in workload_instances[0] and "albert-base-v2_batch2-train" in workload_instances[1]:
            #    print(throughput_of_instances)
        
            #continue  # Skip if any element of the tuple is None
        merged_row = row.tolist() + list(throughput_of_instances)
        merged_rows.append(merged_row)
    print(f"No MPS{targetMPS} throughput  for  {len(no_data)} colocations")

    # Step 6: Create a DataFrame from the merged rows
    throughput_cols = [f"w{i+1}throughput" for i in range(n_combination)]
    merged_df = pd.DataFrame(merged_rows, columns=list(kernel_file.columns) + throughput_cols)
    #drop nan values
    merged_df.dropna(inplace=True)

    #print(f"No MPS{targetMPS} throughput  for the following pairs: {no_data}")
    #Step 7: apply exclusive throughput, sm and mem% to csv_data using base_sm_dict and base_mem_dict
    base_df = pd.read_csv(baselineData)
    #drop row with nan values
    #base_df.dropna(inplace=True)
    #store sm in a dict wuth key as Type from base_df, base_df["sm%"] as value
    if not hotcloud:
        base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
        base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
        base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
        base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
        
        for i in range(n_combination): 
            merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
            merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
            merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
            merged_df[f'w{i+1}memcap'] = merged_df[f'workload{i+1}'].map(base_memcap_dict)
        merged_df.dropna(inplace=True)
        print(merged_df)

    else:
        base_sm_dict = base_df.set_index('Type')['sm%'].to_dict()
        base_mem_dict = base_df.set_index('Type')['mem%'].to_dict()
        #base_memcap_dict = base_df.set_index('Type')['memcap'].to_dict()
        base_throughput_dict  = base_df.set_index('Type')[f'Exclusive{targetMPS}'].to_dict()
        base_CPU_dict = base_df.set_index('Type')['AvgCPU'].to_dict()
        base_MainMem_dict = base_df.set_index('Type')['AvgMem'].to_dict()
        #keep only kernels that we use columns
        #["PCIe read bandwidth", "PCIe write bandwidth", "Long_Kernel",  "ave_Kernel_Length", "long/short_Ratio", "avg_Thread"]
        
        for i in range(n_combination): 
            #merged_df[f'w{i+1}exclusive_throughput'] = merged_df[f'workload{i+1}'].map(base_throughput_dict)
            merged_df[f'w{i+1}sm%'] = merged_df[f'workload{i+1}'].map(base_sm_dict)
            merged_df[f'w{i+1}mem%'] = merged_df[f'workload{i+1}'].map(base_mem_dict)
            merged_df[f'w{i+1}CPU%'] = merged_df[f'workload{i+1}'].map(base_CPU_dict)
            merged_df[f'w{i+1}MainMem%'] = merged_df[f'workload{i+1}'].map(base_MainMem_dict)

        
        merged_df.dropna(inplace=True)

    import os# Get the parent directory from the output file path
    parent_dir = os.path.dirname(outname)
    # Create parent directory if it doesn't exist
    os.makedirs(parent_dir, exist_ok=True)
    #kernel_file.dropna(inplace=True)
    merged_df.to_csv(f"{outname}_targetMPS{targetMPS}.csv", index=False)
    #print noData
   
    print("stage1  training data saved to: ", f"{outname}_targetMPS{targetMPS}.csv")
    return outname


def train_test_split_multiinstance(data,  train_outname, test_outname, random_seed, train_ratio):
    #split train test
    # Load the CSV file into a DataFrame
    df = pd.read_csv(data)
    # Create a dictionary to count the appearances of each workload
    #shuffle df
    training_set = df.sample(frac=train_ratio, random_state=random_seed)
    testing_set = df.drop(training_set.index)



    """
    workload_count = defaultdict(int)

    # Create a list to store the indices of rows to include in the training set
    training_indices = []

    # Iterate over each row in the DataFrame
    #ensure that each workload appears at most n_workload_counts times
    while (all(value < n_workload_counts for value in workload_count.values())):
        for idx, row in df.iterrows():
            workload1 = row['workload1']
            workload2 = row['workload2']
            if workload1 == workload2:
                continue
            # Check if adding this row will exceed the count of 2 for either workload
            if workload_count[workload1] < n_workload_counts or workload_count[workload2] < n_workload_counts:
                training_indices.append(idx)
                workload_count[workload1] += 1
                workload_count[workload2] += 1

    # Create the training set DataFrame
    training_set = df.loc[training_indices]
    testing_set = df.drop(training_indices)
    """
    #get train_outname absoulte path
    #train_outname = os.path.abspath(train_outname)
    print(f"training set save to {os.path.abspath(train_outname)}")
    print(f"testing set save to {os.path.abspath(test_outname)}")
    # Save the training set to a new CSV file
    training_set.to_csv(train_outname, index=False)
    #create testing_set with the rest of the data

    # Save the testing set to a new CSV file
    testing_set.to_csv(test_outname, index=False)


    return  train_outname, test_outname





if __name__ == "__main__":
    RANDOMSEED = 30
    CORRELATION = 0
    TARGET="sum_throughput"
    FILTERED_WORKLOAD = ''
    TEST_WORKLOAD = ''
    """
    filename = '/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/kernel_labels_L2norm.csv'  
    y_all_test = pd.DataFrame()

    df = pd.read_csv(filename)
    df = filter_data(df, FILTERED_WORKLOAD, isbatchThroughput=False)
    X_train, X_test, y_train, y_test, workloads = train_test_custom_split(df, test_workload=TEST_WORKLOAD, target=TARGET,  RANDOMSEED=RANDOMSEED, CORREATION=CORRELATION)
    print(workloads)
    """
    ##############################
    #MULTI INSTANCE
    ##############################
    """
    datafile = get_multiinstance_stage1trainData(baselineData="/home/cc/mlProfiler/tests/mps/analysis/baseline_labels.csv", 
                       shareThroughputData="/home/cc/mlProfiler/tests/mps/analysis/stage2/0730_share3_batch2_share_steps_stage2.csv", 
                       kernelData="/home/cc/mlProfiler/tests/mps/multiinstance/kernel_labels_comb3_batches2.csv",
                       outname="/home/cc/mlProfiler/tests/mps/multiinstance/dataset/total_labels", 
                       targetMPS=100,
                       n_combination=3)
    """

    #split into train/test set
    train_test_split_multiinstance(data="/home/cc/mlProfiler/tests/mps/multiinstance/dataset/total_labels_targetMPS100.csv", 
                                    train_outname="/home/cc/mlProfiler/tests/mps/multiinstance/dataset/training_set.csv", 
                                    test_outname="/home/cc/mlProfiler/tests/mps/multiinstance/dataset/testing.csv", 
                                    random_seed=30, train_ratio=0.5)

