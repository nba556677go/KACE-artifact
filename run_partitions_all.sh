#n_occurs 0 - all data
#n_occurs=(2 3 4 5 6 7 8 9 10)
n_occurs=(0)
#n_occurs=(4 5 6 7)
for n_occur in "${n_occurs[@]}"; do


    #bash rand_run_allworkloads.sh AutoML  0 $n_occur output/run0709_KACE_hotcloud  fe  predict
    #bash rand_run_allworkloads.sh NN  0 $n_occur output/run0709_KACE_hotcloud  fef predict
    #bash rand_run_allworkloads.sh RF  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
    #bash rand_run_allworkloads.sh hotcloud  0 $n_occur output/run0709_KACE_hotcloud  fefe  predict
    #bash partition_allworkloads.sh KACE  0 $n_occur output/partition_exp fefe  predict
    #bash partition_allworkloads.sh NN  0 $n_occur output/partition_exp train  predict
    #bash partition_allworkloads.sh AutoML  0 $n_occur output/partition_exp train  predict
    #bash partition_allworkloads.sh RF  0 $n_occur output/partition_exp train  predict
    bash partition_allworkloads.sh hotcloud  0 $n_occur output/partition_exp train  predict
    
done