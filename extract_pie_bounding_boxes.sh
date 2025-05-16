#!/bin/bash

#SBATCH --job-name=extract_pie_bounding_boxes
#SBATCH --time=10:00:00
#SBATCH --mem=64gb
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --output=output_extract_pie_bounding_boxes_%j.out
#SBATCH --error=output_extract_pie_bounding_boxes_%j.err

# Activate your virtual environment
source /fs/nexus-scratch/eloirghi/STEP/.venv/bin/activate

# Run the script
python /fs/nexus-scratch/eloirghi/STEP/extract_pie_bounding_boxes.py

echo "✅ extract_pie_bounding_boxes.py completed"
