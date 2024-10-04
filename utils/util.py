

# %%
import matplotlib.pyplot as plt 
import csv
import pandas as pd
import os
import sys
"""
Define common functions for all training/inference scripts
"""
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def draw_mem_summary(mem_dict):

    # mem dict example
    #mem_dict = {
    #    1702424247.2902763: {'allocated': 52.9072265625, 'reserved': 514.0, 'state': 'load_model'},
    #    1.0565059185028076: {'allocated': 11996.509765625, 'reserved': 12582.0, 'state': 'fwd'},
    #    4.7603394985198975: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'bwd'},
    #    4.760614395141602: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'optimizer.step'}
    #}

    # Extracting data for plotting
    times = list(mem_dict.keys())
    allocated_memory = [entry['allocated'] for entry in mem_dict.values()]
    reserved_memory = [entry['reserved'] for entry in mem_dict.values()]
    states = [entry['state'] for entry in mem_dict.values()]

    # Creating a marker map for different states
    state_markers = {'load_model': 's', 'fwd': 'x', 'bwd': 'o', 'optimizer.step': '^'}
    markers = [state_markers[state] for state in states]

    # Creating a color map for different states
    state_colors = {'load_model': 'blue', 'fwd': 'green', 'bwd': 'red', 'optimizer.step': 'purple'}
    colors = [state_colors[state] for state in states]
        # Plotting the data
    plt.figure(figsize=(10, 6))

    # Plotting lines and markers separately to include them in the legend
    allocated_line, = plt.plot(times, allocated_memory, linestyle='-', label='Allocated Memory (MB)', color='orange')
    reserved_line, = plt.plot(times, reserved_memory, linestyle='-', label='Reserved Memory (MB)', color='gray')
    allocated_markers = plt.scatter(times, allocated_memory, c=colors)
    reserved_markers = plt.scatter(times, reserved_memory, c=colors)
    state_scatter = plt.scatter(times, allocated_memory, c=colors, label='State Markers')

    # Adding labels and legend
    plt.xlabel('Time (s)')
    plt.ylabel('Memory (MB)')
    plt.title('Memory Usage Over Time')
    plt.legend(handles=[allocated_line, reserved_line,  state_scatter], labels=['Allocated Memory', 'Reserved Memory', 'State Markers'])

    # Show the plot
    plt.savefig('mem_snapshot.jpg')
# %%
def draw_profile_summary(csv, output):

    try:
        # Try reading the file using default UTF-8 encoding
        df = pd.read_csv(csv)
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
    plt.savefig(f"./{output}.jpg")

    print(f"Profile graph stored in./{output}.jpg")
