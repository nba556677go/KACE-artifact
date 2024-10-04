#!/bin/bash

# Check if the file is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <csv_file>"
    exit 1
fi

csv_file="$1"

# Extract the header
header=$(head -n 1 "$csv_file")

# Shuffle the remaining lines
tail -n +2 "$csv_file" | shuf > shuffled_content.csv

# Combine header with shuffled content
echo "$header" > shuffled_output.csv
cat shuffled_content.csv >> shuffled_output.csv

# Clean up temporary shuffled file
rm shuffled_content.csv

echo "Shuffled CSV saved to shuffled_output.csv"
