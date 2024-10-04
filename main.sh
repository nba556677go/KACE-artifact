
#python main.py  -w vit-base-patch16-224_batch2-inf -m /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/trained_models/2462024_09:49:58linear_regression_model_test-vit-base-patch16-224_batch2-inf.pkl

#python main.py -w vit_h_14_batch16-train -m /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/trained_models/2462024_09:47:32linear_regression_model_test-vit_h_14_batch16-train.pkl
#python main.py  -w wav2vec2-base-960h_batch8-inf -m /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/trained_models/2462024_09:51:39linear_regression_model_test-wav2vec2-base-960h_batch8-inf.pkl
#python main.py  -w bert-base-cased_batch2-inf -m /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/training/trained_models/2362024_19:28:58linear_regression_model_test-bert-base-cased_batch2-inf.pkl
model=$1
correlation=$2

test_file="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/MPS100/rand10/testing_set.csv"
train_file="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/dataset/MPS100/rand10/training_set.csv"

#hotcloud
#test_file=tests/mps/analysis/baselines/hotcloud/testing_set_MPS100.csv
#train_file=tests/mps/analysis/baselines/hotcloud/training_set_MPS100.csv

#check if model  and correlation is provided
if [ -z "$correlation" ]
then
    echo "Correlation is not provided"
    exit 1
fi
if [ -z "$model" ]
then
    echo "Model is not provided"
    exit 1
fi



workloads=( 'whisper-large-v2_batch16-inf' 'whisper-large-v2_batch8-inf' 
 'whisper-large-v2_batch2-inf' 'bert-base-cased_batch16-inf'
 'bert-base-cased_batch8-inf' 'vit-base-patch16-224_batch8-inf'
 'vit-base-patch16-224_batch2-inf' 'vit-base-patch16-224_batch16-inf'
 'wav2vec2-base-960h_batch2-inf' 'wav2vec2-base-960h_batch16-inf'
 'wav2vec2-base-960h_batch8-inf' 'bert-base-cased_batch2-inf'
 'vit_h_14_batch8-train' 'vit_h_14_batch16-train'
 'bert-base-cased_batch16-train' 'bert-base-cased_batch8-train'
 'vit_h_14_batch2-train' 'albert-base-v2_batch2-train'
 'albert-base-v2_batch8-train' 'albert-base-v2_batch16-train'
 'bert-base-cased_batch2-train')

#iterate  over the workloads
for workload in "${workloads[@]}"
do
    echo $workload
    python main.py  --test_file $test_file \
    --train_file  $train_file \
    --HPworkload $workload --targetMPS 100 \
    --model $model -corr $correlation \
    --rulebase
    #--model $model -corr 0.2
    #no_exclusive_feat model
    #--model output/trained_models/26-6-2024_03:19:04linear_regression_model.pkl
    #with exclusive_throughput as only feat model
    #--model output/trained_models/26-6-2024_10:31:13linear_regression_model.pkl --correlation 0.8
    
    #--sharedThroughputData /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/stage2/merged0505_0617_0520_0624_share_steps_stage2.csv
    
done
#save  $test_file and $train_file and $model and correlation config in a file
echo testfile=$test_file > output/MPS100/config.txt
echo trainfile=$train_file >> output/MPS100/config.txt
echo model=$model >> output/MPS100/config.txt
echo correlation=$correlation >> output/MPS100/config.txt




#plot average throughput
python output/count_average.py  output/MPS100