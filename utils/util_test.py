from profiler import MLProfiler
from parser import MLParser
import os
import sys
import copy



def draw_mem_summary():

    # Provided data
    import matplotlib.pyplot as plt

    # Provided data
    mem_dict = {
        1702424247.2902763: {'allocated': 52.9072265625, 'reserved': 514.0, 'state': 'load_model'},
        1.0565059185028076: {'allocated': 11996.509765625, 'reserved': 12582.0, 'state': 'fwd'},
        4.7603394985198975: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'bwd'},
        4.760614395141602: {'allocated': 12191.27880859375, 'reserved': 13866.0, 'state': 'optimizer.step'}
    }

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


draw_mem_summary()