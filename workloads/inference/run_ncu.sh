echo quit | sudo nvidia-cuda-mps-control
sudo /usr/local/cuda/bin/ncu  --nvtx --nvtx-include "forward/"  --call-stack --target-processes all --verbose  --set full --csv -o ../../tests/mps/analysis/kernel_profiles/source/ncu/vit-base-patch16-224_batch2-inf /home/cc/miniconda3/bin/python imgclassification-inference.py --model_name google/vit-base-patch16-224  --batch_size 2  --profile_nstep 12 --profile
sudo /usr/local/cuda/bin/ncu  --nvtx --nvtx-include "forward/"  --call-stack --target-processes all --verbose  --set full --csv -o ../../tests/mps/analysis/kernel_profiles/source/ncu/vit-base-patch16-224_batch8-inf /home/cc/miniconda3/bin/python imgclassification-inference.py --model_name google/vit-base-patch16-224  --batch_size 8  --profile_nstep 12 --profile
sudo /usr/local/cuda/bin/ncu  --nvtx --nvtx-include "forward/"  --call-stack --target-processes all --verbose  --set full --csv -o ../../tests/mps/analysis/kernel_profiles/source/ncu/vit-base-patch16-224_batch16-inf /home/cc/miniconda3/bin/python imgclassification-inference.py --model_name google/vit-base-patch16-224  --batch_size 16  --profile_nstep 12 --profile
sudo /usr/local/cuda/bin/ncu  --nvtx --nvtx-include "forward/"  --call-stack --target-processes all --verbose  --set full --csv -o ../../tests/mps/analysis/kernel_profiles/source/ncu/vit-base-patch16-224_batch32-inf /home/cc/miniconda3/bin/python imgclassification-inference.py --model_name google/vit-base-patch16-224  --batch_size 32  --profile_nstep 12 --profile

sudo nvidia-cuda-mps-control -d
cd ../../tests/mps
bash baseline_nollm.sh  ccv100_logs/baseline/inf gef train inference



