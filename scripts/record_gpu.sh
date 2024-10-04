#!/bin/bash

# CSV file to store GPU information
csv_dir=$1
csv_file=$2
# File to store the output of the Python script
workload_output_file="python_output_datawrangle.txt"
outputdir="1datawrangle"

mkdir -p ${csv_dir}

# Add header to the CSV file if it doesn't exist
if [ ! -e "$csv_file" ]; then
    echo "timestamp,power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu" > "${csv_dir}/${csv_file}"
fi

# Get current timestamp
timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")

# Infinite loop to record GPU information and run the Python script
while true; do
    # Get current timestamp
    timestamp=$(date -u "+%Y-%m-%d %H:%M:%S")
    

    
    # Run nvidia-smi, prepend timestamp, and append the result to the CSV file
    #echo -n "$timestamp," >> "${csv_dir}/${csv_file}"
    #nvidia-smi pmon -f "${csv_dir}/smi_pmon.csv" -o T
    nvidia-smi --format=csv --query-gpu=timestamp,power.draw,utilization.gpu,memory.used,memory.total,fan.speed,temperature.gpu | tail -n 1 >> "${csv_dir}/${csv_file}" &
    
    # Sleep for a desired interval (e.g., 1 second)
    sleep 1
done
