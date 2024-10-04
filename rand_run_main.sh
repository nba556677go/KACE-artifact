#bash rand_run_main.sh /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/trained_models/rand_shuffles_corr0_excol_sharedmem/8-7-2024_22\:35\:40_LinearRegression_model-corr0.0_datard10_excol_Static_Shared_Memory.pkl  0 output/dataset/MPS100/rand10/testing_set.csv output/dataset/MPS100/rand10/training_set.csv

#mkdir output/MPS100
#bash rand_run_main.sh /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/output/trained_models/rand_shuffles_corr0_excol_sharedmem/8-7-2024_22\:40\:47_LinearRegression_model-corr0.0_datard50_excol_Static_Shared_Memory.pkl 0  output/dataset/MPS100/rand50/testing_set.csv output/dataset/MPS100/rand50/training_set.csv
#mv output/MPS100 output/rand_run_nosharedmem/rand50
modeltype=$1
correlation=$2
n_occur=$3
outdir=$4
istrain=$5
ispredict=$6



#check if model  and correlation is provided
if [ -z "$correlation" ]
then
    echo "Correlation is not provided"
    exit 1
fi
if [ -z "$modeltype" ]
then
    echo "Model is not provided"
    exit 1
fi
#model could only be KACE or hotcloud, AutoML, NN, RF


#if [ "$modeltype" != "KACE" ] && [ "$modeltype" != "hotcloud" ] && [ "$modeltype" != "AutoML" ]
#then
#    echo "Model should be either KACE or hotcloud"
#    exit 1
#fi

#modify if needed:
############################################
#EXCOL="_Static Shared Memory"
EXCOL=""
#for splitting
root_dataset_dir="output/dataset/MPS100/n_occur_${n_occur}"
if [ "$modeltype" == "hotcloud" ]
then
    full_data_path="/Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/baselines/hotcloud/hotcloud_combined_noexclu_throughput_targetMPS100.csv"
else
    #KACE
    full_data_path="output/dataset/MPS100/MPS100_kernel_labels_targetMPS100.csv"
fi

test_file="testing_set.csv"
train_file="training_set.csv"
targetMPS=100

############################################

#create random splits of different 





echo "generate random splits using seed 10-50 and train model"
seeds=(10 20 30 40 50)
#split and train if istrain is set
if [ "$istrain" == "train" ]
then
    for seed in "${seeds[@]}"
    do
        echo "seed $seed"
        dataset_dir="$root_dataset_dir/rand${seed}/$modeltype"
        outdir_rand="$outdir/n_occur_${n_occur}/rand${seed}/$modeltype"
        mkdir -p $dataset_dir
        mkdir -p $outdir_rand
        #if data is not generated, generate data
        if [ ! -f "$dataset_dir/$test_file" ] || [ ! -f "$dataset_dir/$train_file" ]
        then
            echo "generate data for $dataset_dir..."
            python main.py  --test_file $dataset_dir/$test_file \
            --train_file  $dataset_dir/$train_file \
            -rd $seed --targetMPS $targetMPS \
            -d $full_data_path \
            --n_occur $n_occur \
            -c
            echo "data saved in $dataset_dir"
        fi

        echo "train model with $dataset_dir/$train_file..."
        python main.py --test_file $dataset_dir/$test_file --train_file $dataset_dir/$train_file \
        -t $targetMPS -rd $seed  --train -corr $correlation \
        --output_dir $outdir_rand --modeltype $modeltype
    done
fi


#hotcloud
#test_file=tests/mps/analysis/baselines/hotcloud/testing_set_MPS100.csv
#train_file=tests/mps/analysis/baselines/hotcloud/training_set_MPS100.csv


if [ "$ispredict" == "predict" ]
then
############################################
#Predict throughput for different workloads
    echo "start predicting throughput for different workloads"


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

    for seed in "${seeds[@]}";
    do

        echo "seed $seed"
        dataset_dir="$root_dataset_dir/rand${seed}/$modeltype"
        outdir_rand="$outdir/n_occur_${n_occur}/rand${seed}/$modeltype"
        outdir_pred="$outdir_rand/predicts"
        mkdir -p $outdir_pred
        #find model
        if [ $modeltype != "AutoML" ]
        then
            model_file=$(find $outdir_rand -name "*$modeltype*.pkl")
        else
            model_file=$(find $outdir_rand -type f -name "*AutoML*" ! -name "*.txt")
        fi
        #model_file=$(find $outdir_rand -name "*$modeltype*.pkl")
        #raise error and exit if model is not found or multiple models are found
        if [ -z "$model_file" ]
        then
            echo "Model $modeltype not found in $outdir_rand"
            exit 1
        fi
        if [ $(echo $model_file | wc -l) -gt 1 ]
        then
            echo "Multiple models found in $outdir_rand"
            exit 1
        fi

        echo "using model $model_file to predict..."
        #iterate  over the workloads
        for workload in "${workloads[@]}"
        do
            echo "processing $workload..."
            python main.py  --test_file $dataset_dir/$test_file \
            --train_file  $dataset_dir/$train_file \
            --HPworkload $workload --targetMPS 100 \
            --model $model_file -corr $correlation \
            -rd $seed --output_dir $outdir_pred \
            --rulebase --modeltype $modeltype
            #--model $model -corr 0.2
            #no_exclusive_feat model
            #--model output/trained_models/26-6-2024_03:19:04linear_regression_model.pkl
            #with exclusive_throughput as only feat model
            #--model output/trained_models/2d6-6-2024_10:31:13linear_regression_model.pkl --correlation 0.8
            
            #--sharedThroughputData /Users/bing/Library/CloudStorage/OneDrive-StonyBrookUniversity/SBU/mlsys/mlProfiler/tests/mps/analysis/stage2/merged0505_0617_0520_0624_share_steps_stage2.csv
            
        done

        #save  $test_file and $train_file and $model and correlation config in a file
        echo testfile=$dataset_dir/$test_file >  $outdir_pred/config.txt
        echo trainfile=$dataset_dir/$train_file >>  $outdir_pred/config.txt
        echo model=$model_file >>  $outdir_pred/config.txt
        echo correlation=$correlation >>  $outdir_pred/config.txt
        echo train_n_occur=$n_occur >>  $outdir_pred/config.txt
        #plot average throughput
        python output/count_average.py   $outdir_pred $modeltype
        # Check if the Python script exited with a non-zero status
        if [ $? -ne 0 ]; then
            echo "An error occurred while running thecount_average.py script. Exiting..."
            echo "current output directory is $outdir_pred"
            # Optionally, exit the script if the error is critical
            exit 1
        fi
    done
fi





