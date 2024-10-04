# KACE: Kernel-Aware Colocation for Efficient GPU Sharing
This is the repo for SoCC'24 paper - KACE: Kernel-Aware Colocation for Efficient GPU Sharing. 
## Table of contents 
- [Abstract](#abstract)
- [Hardware test environment](#hardware-test-environment)
- [Figure reproduce](#figure-reproduce)
- [Profile offline kernel and system metrics](#profile-offline-kernel-and-system-metrics)
- [Generate colocate workload combinations with kernel metrics](#generate-colocate-workload-combinations-with-kernel-metrics)
- [Get shared throughput from kernel metric combinations](#get-shared-throughput-from-kernel-metric-combinations)
- [Construct total feature set with shared throughput as labels](#construct-total-feature-set-with-shared-throughput-as-labels)
- [Split train/test](#split-train-test)
- [predict all throughput without partitions figure 2](#predict-all-throughput-without-partitions-figure-2)
- [predict throughput with partitions of workloads figure 3 4](#predict-throughput-with-partitions-of-workloads-figure-3-4)
- [Predict unseen partition](#predict-unseen-partition)
## Abstract
GPU spatial sharing among jobs is an effective approach to increase resource utilization and reduce the monetary and environmental costs of running deep learning workloads. While hardware support for GPU spatial sharing already exists, accurately predicting GPU interference between colocated workloads remains a concern. This makes it challenging to improve GPU utilization by sharing the GPU between workloads without severely impacting their performance. Existing approaches to identify and mitigate GPU interference often require extensive profiling and/or hardware modifications, making them difficult to deploy in practice.

This paper presents KACE, a lightweight, prediction-based approach to effectively colocate workloads on a given GPU. KACE adequately predicts colocation interference via exclusive kernel metrics using limited training data and minimal training time, eliminating the need for extensive online profiling of each new workload colocation. Experimental results using various training and inference workloads show that KACE outperforms existing rule-based and prediction-based policies by 16% and 11%, on average, respectively, and is within 10% of the performance achieved by an offline-optimal oracle policy.


## Hardware test environment
```
OS: Ubuntu 20.04 LTS
GPU model: NVIDIA Tesla V100 32 GB
Driver: 545.23.06
CUDA version: 12.3
Docker: 27.3.1
```

## Figure reproduce
To reproduce figures in the paper, check out reproduce_figures/ for reproduce instructions

## Profile offline kernel and system metrics
1. check profiling directory for instructions on profiling system and kernel metrics. 
2. get exclusive throughput
```
cd scripts
bash baseline.sh
```
3. parse baseline metrics
```
python utils/parse_baseline_metrics.py
```
## generate colocate workload combinations with kernel metrics

```
python profiling/kernel_metrics/get_all_kernel_combinations.py
```




## get shared throughput from kernel metric combinations
1. execute shared throughput 
```
//MODIFY csv input of multiinstance based on nimber of combinations in the script
bash scripts/multiinstance/multiinstance.sh
```
2. parse shared_throughput and save
```
#configure input path before execution
python output/shared_throughput/parse_shared_throughput.py shared_throughput dir output_predix num_combinations
```

### Construct total feature set with shared throughput as labels
```
#CHECK OUTNAME,KERNELFILE, BASELINE_FILEfirst!!!!!!
ex. python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 --processStage1Data -sd [shared throughut data] --pred_multiinstance -comb [num_combinations] --modeltype KACE

ex. python main.py  --test_file output/dataset/testing_set.csv --train_file output/dataset/training_set.csv  -t 100 --processStage1Data -sd output/shared_throughput/0730_share3_batch2-8_share_steps_stage2.csv --pred_multiinstance -comb 3 --modeltype KACE
```

### split train test
```
python splitData.py -d $full_data_path \
             --train_file  $dataset_dir/$train_file \
             --test_file $dataset_dir/$test_file \
            -rd $seed --train_ratio $n_occur
```

### predict all throughput without partitions figure 2
```
#CHANGE predict flags to predict_all_throughput
bash run_train_splits.sh 
```
### predict throughput with partitions of workloads figure 3 4
```
#CHANGE predict flags to predict_by_workload
bash scripts/multiinstance/run_train_splits.sh 
```

### predict unseen partition
```
bash scripts/multiinstance/run_unseen_partitions.sh
```
