import torch
import os
import pandas as pd
import sys
import math

#nvidia-smi profile
from collections import defaultdict
import re
import subprocess
import select
import argparse
from termcolor import colored

#write file
import csv
import matplotlib.pyplot as plt 
import copy


class MLProfiler():
    def __init__(self, args):
        self.args = args
        #record training time for each epoch
        self.recordTimes = []
        #nvidia-smi info
        self.SMIinfo = defaultdict(list) 
        self.memDict = {}
        print(f'[Profile] profile mode on')

    def getModelMemory(self):
        print(f"[Profile] model used maximum of {torch.cuda.max_memory_allocated(device=None) / (1024**2)} MB")
    
    def countModelParameters(self, model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        print(f'model size count by param and buffer= {size_all_mb:.3f}MB')
        print(f"[Profile] torch.cuda.memory_allocated should be the same as model parameters={torch.cuda.memory_allocated(device=None)/(1024**2):.3f}MB")

    def getMemStats(self, ts, state):
         #Profile mode - 
        # enable memory history, which will
        # add tracebacks and event history to snapshots
        allocated, reserved = torch.cuda.max_memory_allocated(device=None) / (1024**2), torch.cuda.max_memory_reserved(device=None) / (1024**2)
        print(f"[Profile] allocated maximum of {allocated} MB")
        print(f"[Profile] reserved maximum of {reserved} MB")
        self.memDict[ts] = {'allocated' :  allocated, 'reserved' : reserved, 'state' : state}
        return allocated, reserved
        #print(f"[Profile] run mem stats: {torch.cuda.memory.memory_stats(device=self.args.device)}")
        #print(f"[Profile] get mem snapshot: {torch.cuda.memory_snapshot()}")
    def drawMemSummary(self, output):
        import itertools
        # mem dict example
        #mem_dict = {
        #    1702424247.2902763: {'allocated': 52.9072265625, 'reserved': 514.0, 'state': 'load_model'},
        #    1.0565059185028076: {'allocated': 11996.509765625, 'reserved': 12582.0, 'state': 'fwd'},
        #    4.7603394985198975: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'bwd'},
        #    4.760614395141602: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'optimizer.step'}
        #}
        print(self.memDict)
        # Extracting data for plotting
        times = list(self.memDict.keys())
        allocated_memory = [entry['allocated'] for entry in self.memDict.values()]
        reserved_memory = [entry['reserved'] for entry in self.memDict.values()]
        states = [entry['state'] for entry in self.memDict.values()]

        # Creating a marker map for different states
        #state_markers = {'load_model': 's', 'fwd': 'x', 'bwd': 'o', 'optimizer.step': '^'}
        #markers = [state_markers[state] for state in states]
        colors =  ['b', 'r', 'g', 'c', 'm', 'k']
        #print(self.memDict.values())
        print(set(states))
        state_colors = { entry: colors[i] for i, entry in enumerate(set(states))}
        # Creating a color map for different states
        #state_colors = {'load_model': 'blue', 'fwd': 'green', 'bwd': 'red', 'optimizer.step': 'purple'}
        
        # Plotting the data
        plt.figure(figsize=(10, 6))




        # Plotting lines and markers separately to include them in the legend
        allocated_line, = plt.plot(times, allocated_memory, linestyle='-', label='Allocated Memory (MB)', color='orange')
        reserved_line, = plt.plot(times, reserved_memory, linestyle='-', label='Reserved Memory (MB)', color='gray')
        allocated_markers = plt.scatter(times, allocated_memory, c=[state_colors[state] for state in states])
        reserved_markers = plt.scatter(times, reserved_memory, c=[state_colors[state] for state in states])
        # Adjust color of lines and markers to match state_colors dictionary

        #allocated_markers = plt.scatter(times, allocated_memory, c=colors)
        # Adding labels and legend
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (MB)')
        plt.title(f'Memory Usage Over Time - {output[:-4]}')
        plt.legend(handles=[allocated_line, reserved_line], labels=['Allocated Memory', 'Reserved Memory'])
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colorbar import Colorbar
        # Create a colormap object
        cmap = plt.cm.tab20
        # Map states to color indices
        state_indices = {state: i for i, state in enumerate(set(states))}
        # Create a ScalarMappable object
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(min(state_indices.values()), max(state_indices.values())))
        sm.set_array([])  # Empty array to set the colorbar range
        # Add the colorbar to the plot
        cax = plt.axes([0.92, 0.2, 0.02, 0.6])  # Adjust position as needed
        sm_colorbar = Colorbar(cax, sm, label='State')
        # Label each colorbar tick with the corresponding state
        cax.set_yticklabels([state for state, _ in state_indices.items()])
        """
        # Show the plot
        plt.savefig(output)

    def addEpochTime(self, epoch_time):
        self.recordTimes.append(epoch_time)

    def saveEarlyStop(self,taskName ,taskGPUinfo, total_training_time_per_epoch, fields = ["JOB", "GPU_MEMORY", "GPU_COMPTIME", "CPU_COMPTIME"]):
        """
        TODO - save the last estimated completion time to file to configure K8S pod
        output - 
            {GPU_MEMORY: "1747", GPU_COMPTIME: "120", CPU_COMPTIME: "200"}
        """
        total_training_time = total_training_time_per_epoch * self.args.n_epoch
        fields.extend(list(vars(self.args).keys()))
        output = f"job_env_run_{self.args.profile_run}.csv"
        # handle exception for single epoch
        taskGPUinfo = list(taskGPUinfo.values())[-1][-1] 
        filename = f"{sys.argv[0][:-3]}_epochs_{self.args.n_epoch}.csv"
        profile_csv = os.path.join(os.getcwd(), self.args.profile_output_dir, output)
        os.makedirs(os.path.dirname(profile_csv), exist_ok=True)

        #first write - write title
        if not os.path.isfile(profile_csv):
            with open(profile_csv, "w") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(fields)
                
        #continue to write config...
        with open(profile_csv, "a") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([taskName, taskGPUinfo['gpu_mem'][:-3], int(total_training_time), int(total_training_time)*12,  *list(vars(self.args).values())])#gpu_mem - strip MiB
        print(f"[Profile] profiled results for job config stored at {profile_csv}")
            
        
        
    def saveAllEpochEstimatedTime(self, total_training_time, training_time_per_epoch):
       
        filename = f"epochs_{self.args.n_epoch}_bert_{self.args.device}.csv"
        fields = ["profile_epochs", "total epochs", "profiled_ave_time_per_epoch(s)", "true_ave_time_per_epoch(s)", "total_time_diff(s)", "diff_percent(%)"]
        profile_epochs  = [i for i in range(1, self.args.n_epoch+1, 2)]
        
        profile_csv = os.path.join(os.getcwd(), self.args.profile_output_dir, filename)
        os.makedirs(os.path.dirname(profile_csv), exist_ok=True)
        with open(profile_csv, "w") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
        # skip 1st epoch since it includes startup time
            for PROFILE_EPOCH in profile_epochs:
                estimated_training_time_per_epoch = sum(self.recordTimes[:PROFILE_EPOCH])/PROFILE_EPOCH
                m, s = divmod( estimated_training_time_per_epoch , 60)
                h, m = divmod(m, 60)
                print(f"[Profile] average time for profiling Epochs 1 to {PROFILE_EPOCH} : {estimated_training_time_per_epoch:.2f}s ({h:f}h{ m:02f}m {s:02f}s) ")

                #estimated job completion time
                estimated_time_for_job = estimated_training_time_per_epoch * self.args.n_epoch
                m, s = divmod(estimated_time_for_job , 60)
                h, m = divmod(m, 60)
                print(f"[Profile] Estimated Training time for all epochs {h:f}h{ m:02f}m {s:02f}s")
                print(f"[Profile] diff in estimated vs actual training time : {estimated_time_for_job - total_training_time:.2f}s, diff percentage: {((estimated_time_for_job - total_training_time)/ total_training_time) *100}%")
                # torch.cuda.memory._dump_snapshot("resnet50snapshot.pickle")
                #print(f"[Profile] torch memory stats: {torch.cuda.memory.memory_stats(device=device)}")
                #write results into csv

                data = [PROFILE_EPOCH, self.args.n_epoch, estimated_training_time_per_epoch, training_time_per_epoch, estimated_time_for_job - total_training_time, ((estimated_time_for_job - total_training_time)/ total_training_time) *100 ]
            
                csvwriter.writerow(data)
        print(f"Profile results stored in {self.args.profile_output_dir}/{filename}")
        return filename


    def getSMIinfobyTask(self, task_command):
        #get nvidia-smi info
        MEMORY_FREE_RATIO = 0.05
        MEMORY_MODERATE_RATIO = 0.9
        GPU_FREE_RATIO = 0.05
        GPU_MODERATE_RATIO = 0.75
        TASK_PID = [] #select PID when reading nvidia-smi info

        # parse the command length argument
        command_length = 20
        color = False
        fake_ps = None

        # for testing, the stdin can be provided in a file
        fake_stdin_path = os.getenv("FAKE_STDIN_PATH", None)
        stdin_lines = []
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            stdin_lines = sys.stdin.readlines()

        if fake_stdin_path is not None:
            with open(fake_stdin_path, 'rt') as f:
                lines = f.readlines()
        elif stdin_lines:
            lines = stdin_lines
        else:
            ps_call = subprocess.run('nvidia-smi', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if ps_call.returncode != 0:
                print('nvidia-smi exited with error code {}:'.format(ps_call.returncode))
                print(ps_call.stdout.decode() + ps_call.stderr.decode())
                sys.exit()
            lines_proc = ps_call.stdout.decode().split("\n")
            lines = [line + '\n' for line in lines_proc[:-1]]
            lines += lines_proc[-1]


        def colorize(_lines):
            for j in range(len(_lines)):
                line = _lines[j]
                m = re.match(r"\| (?:N/A|..%)\s+[0-9]{2,3}C.*\s([0-9]+)MiB\s+/\s+([0-9]+)MiB.*\s([0-9]+)%", line)
                if m is not None:
                    used_mem = int(m.group(1))
                    total_mem = int(m.group(2))
                    gpu_util = int(m.group(3)) / 100.0
                    mem_util = used_mem / float(total_mem)

                    is_moderate = False
                    is_high = gpu_util >= GPU_MODERATE_RATIO or mem_util >= MEMORY_MODERATE_RATIO
                    if not is_high:
                        is_moderate = gpu_util >= GPU_FREE_RATIO or mem_util >= MEMORY_FREE_RATIO

                    c = 'red' if is_high else ('yellow' if is_moderate else 'green')
                    _lines[j] = colored(_lines[j], c)
                    _lines[j-1] = colored(_lines[j-1], c)

            return _lines


        lines_to_print = []
        is_new_format = False
        # Copy the utilization upper part verbatim
        for i in range(len(lines)):
            if not lines[i].startswith("| Processes:"):
                lines_to_print.append(lines[i].rstrip())
            else:
                while not lines[i].startswith("|===="):
                    m = re.search(r'GPU\s*GI\s*CI', lines[i])
                    if m is not None:
                        is_new_format = True
                    i += 1
                i += 1
                break

        if color:
            lines_to_print = colorize(lines_to_print)

        # we print all but the last line which is the +---+ separator
        for line in lines_to_print[:-1]:
            print(line)

        no_running_process = "No running processes found"
        if no_running_process in lines[i] or lines[i].startswith("+--"):
            #print(lines[-1].strip())
            #print("| " + no_running_process + " " * (73 - len(no_running_process)) + "   |")
            # Issue #9, running inside docker and seeing no processes
            if lines[i].startswith("+--"):
                print("| If you're running in a container, you'll only see processes running inside. |")
            #print(lines[-1])
            sys.exit()

        # Parse the PIDs from the lower part
        gpu_num = []
        pid = []
        gpu_mem = []
        user = []
        cpu = []
        mem = []
        time = []
        command = []

        gpu_num_idx = 1
        pid_idx = 2 if not is_new_format else 4
        gpu_mem_idx = -3

        while not lines[i].startswith("+--"):
            if "Not Supported" in lines[i]:
                i += 1
                continue
            line = lines[i]
            line = re.split(r'\s+', line)
            gpu_num.append(line[gpu_num_idx])
            pid.append(line[pid_idx])
            gpu_mem.append(line[gpu_mem_idx])
            user.append("")
            cpu.append("")
            mem.append("")
            time.append("")
            command.append("")
            i += 1
        #print(pid)

        if fake_ps is None:
            # Query the PIDs using ps
            ps_format = "pid,user,%cpu,%mem,etime,command"
            ps_call = subprocess.run(["ps", "-o", ps_format, "-p", ",".join(pid)], stdout=subprocess.PIPE)
            processes = ps_call.stdout.decode().split("\n")
        else:
            with open(fake_ps, 'r') as f:
                processes = f.readlines()

        # Parse ps output
        for line in processes:
            if line.strip().startswith("PID") or len(line) == 0:
                continue
            parts = re.split(r'\s+', line.strip(), 5)
            # idx = pid.index(parts[0])
            for idx in filter(lambda p: pid[p] == parts[0], range(len(pid))):
                user[idx] = parts[1]
                cpu[idx] = parts[2]
                mem[idx] = parts[3]
                time[idx] = parts[4] if "-" not in parts[4] else parts[4].split("-")[0] + " days"
                command[idx] = parts[5]


        """
        TODO minimize gpu mem estimation after switching to clean environment
        """
        for i in range(len(pid)):
            self.SMIinfo[pid[i]].append({
                                    "gpu_num" : gpu_num[i],
                                    "user" : user[i],
                                    "gpu_mem" : gpu_mem[i],
                                    "cpu": cpu[i],
                                    "mem": mem[i],
                                    "time": time[i],
                                    "command": command[i]
                                    
            })
            # find pid for queried task+model+dataset from command
            
            if ' '.join(task_command) in command[i]:             
                #may have multiple process running same task
                TASK_PID.append(pid[i])
            
       # print(f"PID: {i}, nvidia-smi info - {self.SMIinfo[TASK_PID[i]]}") for pid in TASKPID 
        results = {pid : self.SMIinfo[pid] for pid in TASK_PID }
        return results if len(TASK_PID) > 0 else print(f"ERROR - cannot find PID with {' '.join(task_command)}")
    
    def draw_summary(self, csvfile):
        
        try:
        # Try reading the file using default UTF-8 encoding
            df = pd.read_csv(os.path.join(".", f"{self.args.profile_output_dir}", csvfile))
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, try reading the file using UTF-16 encoding with tab separator
                df = pd.read_csv(csv, sep=',', encoding='utf-16')
            except Exception as e:
                print(f"Could not read file {csv} because of error: {e}")
        except Exception as e:
            print(f"Could not read file {csv} because of error: {e}")



        # Define a list of colors to use for each profile_epochs group
        import itertools
        colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        # Create a dictionary to store colors for each unique profile_epochs
        color_dict = {}

        # Iterate through the data and plot the lines with the same color for each profile_epochs group
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        #fig.set_xlabel('Total Epochs')
        fig.text(0.5, 0.04, 'Total epochs', ha='center')
        ax1.plot(df["profile_epochs"], df['profiled_ave_time_per_epoch(s)'],
                    label=f'Profiled time per epoch', linestyle='--', marker='o', markersize=3)
        ax1.plot(df["profile_epochs"], df['true_ave_time_per_epoch(s)'],
                    label=f'True training time per epoch', marker='o', markersize=3)

        # Set labels and legend
        #ax1.set_xlabel('Total Epochs')
        ax1.set_ylabel('Time per epoch(s)')
        ax1.set_title("Fig1 average profiled time")
        ax1.legend()

        # Show the grid
        ax1.grid()


        #plot diff

        ax2.plot(df["profile_epochs"], df["total_time_diff(s)"],
                    label=f'Total estimated completion time - actual training time', linestyle='--', marker='o', markersize=3)
        #ax2_1 = ax2.twinx()

        # Set labels and legend
        ax2.set_ylabel('total diff time (s)')
        ax2.set_title("Fig2 Total diff time")
        ax2.legend()

        # Show the grid
        ax2.grid()

        #fig3 - diff graph

        #ax2_1.set_ylim(-2,2)
        ax3.plot(df["profile_epochs"], df['diff_percent(%)'],
                    label=f'Diff time percentage (%) ', marker='o', linestyle='--', markersize=3)

        # Set labels and legend
        ax3.set_ylabel('total diff time %')
        #ax2_1.set_ylabel('diff time (%)')

        ax3.set_title("Fig3 Diff time percentage (%)")
        ax3.legend()
        #ax2_1.legend()

        # Show the grid
        ax3.grid()

        # Display the plot
        plt.savefig(f"./{self.args.profile_output_dir}/{csvfile}.jpg")

        print(f"Profile graph stored in {self.args.profile_output_dir}/{csvfile}.jpg")

