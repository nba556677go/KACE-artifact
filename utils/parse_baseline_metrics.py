
import sys
from parse_util import *
output_file = f"baseline_metrics.csv"
# Save the average sm and memory to a new CSV file
dirname = sys.argv[1]

BEtypes = ["train", "inf"]  # or "train"
result_dict = defaultdict(dict)
for BEtype in BEtypes:
    
#save_to_csv(avg_sm, avg_mem, filename)
    result = get_all_base_results(f"{dirname}/{BEtype}", BEtype)
    #update result_dict with result
    result_dict.update(result)
save_baseline_to_csv(result_dict, f"baseline_metrics.csv")
#save result_dict to csv

    