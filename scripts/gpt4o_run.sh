#!/bin/bash

# Initial start_idx
start_idx=645

# Path to result file
result_file="../../datasets/results/gpt4o_final/curr_final_2.json"

# Loop until start_idx reaches 1000
while [ $start_idx -lt 1000 ]
do
    # Run the Python script
    python3 debating_with_gpt4o.py --start_idx $start_idx --result_file "$result_file"
    
    # Check if the Python script exited successfully
    if [ $? -eq 0 ]
    then
        echo "Python script completed successfully."
    else
        echo "Python script failed. Exiting loop."
        break
    fi
    
    # Increment start_idx
    start_idx=$((start_idx + 1))
    
    echo "Next start_idx: $start_idx"
done

echo "Script execution completed."