Please not use containered workload when profiling, since Nsight compute does not support. Instead, run DL workload directly with host environment 
```
cd ..
pip install -r requirements.txt
```

# Profile kernel metrics
We use Nsight Compute to profile kernel metrics and filter out useful metrics with csv parsing
```
cd kernel metrics/output_KACE
//CHECK: argument path in the following script
//profile inference workload
bash total_ncu_inf.sh
//profile training workload
bash total_ncu_train.sh
```
## Xu et al baseline kernel metric collection
Once Nsight Compute source files are obtained, parse hotcloud required metrics 
```
cd kernel metrics/output_KACE
bash hotcloud_parse_ncu_kernels.sh
```

# Profile system metrics
parse system metrics from collected logs
```
cd system_metrics/KACE
python parse_util.py
```
## Xu et al baseline system metrics parsing
```
cd system_metrics/hotcloud
python hotcloud_parse.py
```
