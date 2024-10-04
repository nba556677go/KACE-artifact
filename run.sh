
n_occurs=(2 3 4 5 6 7 8 9 10)
#n_occurs=(2 3 4 5 6 7)
for n_occur in "${n_occurs[@]}"; do
    bash rand_run_main.sh KACE  0 $n_occur output/run0709_KACE_hotcloud  train  predict
    bash rand_run_main.sh AutoML  0 $n_occur output/run0709_KACE_hotcloud  train  predict
    bash rand_run_main.sh NN  0 $n_occur output/run0709_KACE_hotcloud  train  predict
    bash rand_run_main.sh RF  0 $n_occur output/run0709_KACE_hotcloud  train  predict
    bash rand_run_main.sh hotcloud  0 $n_occur output/run0709_KACE_hotcloud  train  predict
done