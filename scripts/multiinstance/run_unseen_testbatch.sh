n_occur=0
#fefe
#for n_occur in "${n_occurs[@]}"; do
#bash unseen_partitions.sh KACE 0 $n_occur output/run0730_KACE_batch2-8  fefe  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh AutoML 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh NN 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh RF 0 $n_occur output/run0730_KACE_batch2-8  train  predict
#bash unseen_partitions.sh KACE 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0906_comb4_KACE_batch2-8  train  predict
bash unseen_testbatch.sh KACE 0 $n_occur output/run0909_comb4_KACE_testbatch4_batch2-8  train  predict
#bash unseen_partitions.sh hotcloud 0 $n_occur output/run0907_comb4_KACE_batch2-8  train  predict
