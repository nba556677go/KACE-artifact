
#n_occurs=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#n_occur not used in unseen
n_occur=0
#trainmodel is fixed while predicting with partitions
#predict_by_workload - predict with each target workload set
#predict - predict with all workload combinations excludes from training set
#for n_occur in "${n_occurs[@]}"; do
#bash unseen_partitions.sh KACE 0 $n_occur output/run0730_KACE_batch2-8  fefe  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh AutoML 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh NN 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh RF 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh KACE 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
bash unseen_partition_batch.sh KACE 0 $n_occur output/run0906_comb4_KACE_batch2-8_partition_train_batch2  fefef  predict_by_workload

