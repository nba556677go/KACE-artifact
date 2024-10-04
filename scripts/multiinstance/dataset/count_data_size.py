import os
import pandas as pd
import sys
# Define the root directory where the files are located
root_dir = sys.argv[1]

# Initialize lists to store the number of lines in each file
training_counts = []
testing_counts = []

# Walk through the directory tree
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        if file == 'training_set.csv':
            # Read the training file and count the lines
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)
            training_counts.append(len(df))

        elif file == 'testing_set.csv':
            # Read the testing file and count the lines
            file_path = os.path.join(subdir, file)
            df = pd.read_csv(file_path)
            testing_counts.append(len(df))
print(training_counts)
print(testing_counts)
# Calculate the averages
average_training_lines = sum(training_counts) / len(training_counts) if training_counts else 0
average_testing_lines = sum(testing_counts) / len(testing_counts) if testing_counts else 0

# Print the results
print(f"Average number of lines in training sets: {average_training_lines}")
print(f"Average number of lines in testing sets: {average_testing_lines}")
