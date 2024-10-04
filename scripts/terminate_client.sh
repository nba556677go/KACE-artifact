#!/bin/bash

# Get the output of "echo ps | nvidia-cuda-mps-control" and store it in a variable
mps_ps_output=$(echo ps | nvidia-cuda-mps-control)

# Check if the output contains any lines
if [ -z "$mps_ps_output" ]; then
    echo "No MPS server processes found."
    #echo quit | sudo nvidia-cuda-mps-control
    #sleep 1
    #sudo nvidia-cuda-mps-control -d
    exit 1
fi

echo "force shutdown mps server..."
echo shutdown_server  $(echo  get_server_list | nvidia-cuda-mps-control) -f | sudo nvidia-cuda-mps-control
# Loop through each entry in the output to extract the server and PID of the client process
echo "$mps_ps_output" | awk 'NR>1' | while read -r line; do
    server=$(echo "$line" | awk '{print $3}')
    pid=$(echo "$line" | awk '{print $1}')

    # Check if server and PID are not empty
    if [ -n "$server" ] && [ -n "$pid" ]; then
        # Terminate the client using the "terminate_client" command
        echo "Terminating client with PID $pid on server $server..."
        echo "echo terminate_client $server $pid | nvidia-cuda-mps-control"
        echo terminate_client "$server" "$pid" | nvidia-cuda-mps-control
        sleep 1
        sudo kill -9 "$pid"
    fi
done
#sleep 2
#echo quit | sudo nvidia-cuda-mps-control
sleep 1
#sudo nvidia-cuda-mps-control -d
sleep 1