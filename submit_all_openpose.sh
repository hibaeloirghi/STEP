#!/bin/bash

input_list="/fs/nexus-scratch/eloirghi/STEP/folder_list.txt"
output_root="/fs/nexus-scratch/eloirghi/STEP/openpose_output"
max_jobs=5
sleep_time=120

while read ped_dir; do
    ped_name=$(basename "$ped_dir")
    json_dir="$output_root/$ped_name/json"
    
    # Skip if already processed
    if [ -d "$json_dir" ] && ls "$json_dir"/*.json 1>/dev/null 2>&1; then
        echo "Skipping $ped_name: already processed."
        continue
    fi

    # Throttle job submission
    while true; do
        num_jobs=$(squeue -u "$USER" | wc -l)
        if [ "$num_jobs" -lt "$max_jobs" ]; then
            sbatch --export=PED_DIR="$ped_dir" /fs/nexus-scratch/eloirghi/STEP/run_openpose_single.sbatch
            echo "Submitted job for $ped_name (current jobs: $num_jobs)"
            break
        else
            echo "Waiting: $num_jobs jobs in queue, sleeping $sleep_time seconds..."
            sleep "$sleep_time"
        fi
    done
done < "$input_list"
