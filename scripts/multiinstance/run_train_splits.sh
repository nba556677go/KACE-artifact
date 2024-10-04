
#n_occurs=(0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1)
n_occurs=(0.7 0.6 0.5 0.4 0.3 0.2 0.1)
#n_occurs=(0.9)
#n_occurs=(8 9 10)
#n_occurs=(4 5 6 7)
for n_occur in "${n_occurs[@]}"; do


    #bash rand_run_allworkloads.sh AutoML  0 $n_occur output/run0709_KACE_hotcloud  fe  predict
    #bash rand_run_allworkloads.sh NN  0 $n_occur output/run0709_KACE_hotcloud  fef predict
    #bash rand_run_allworkloads.sh RF  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
    #bash train_splits.sh KACE  0 $n_occur output/run0730_KACE_batch2  train predict
    #bash train_splits.sh KACE 0 $n_occur output/run0730_KACE_batch2-8  fefe predict
    bash train_splits.sh AutoML 0 $n_occur output/run0730_KACE_batch2-8  train predict_all_throughput
    #bash train_splits.sh NN 0 $n_occur output/run0730_KACE_batch2-8  wefhuehf predict
    #bash train_splits.sh RF 0 $n_occur output/run0730_KACE_batch2-8  fsefe predict
    #bash rand_run_allworkloads.sh hotcloud  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
done