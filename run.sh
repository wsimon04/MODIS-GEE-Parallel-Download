#!/bin/bash
#SBATCH -c 8                                 # Number of CPU cores
#SBATCH --mem=16G                            # Total memory
#SBATCH -p cpu                               # Partition
#SBATCH -t 168:00:00                         # Set time
#SBATCH --qos=long                           # Use QoS 'long' for extended runtime
#SBATCH -o slurm-group_%x_%j.out             # Output log

# Load Conda environment
module load conda/latest
source /insert/path/here
conda activate replace_with_your_env_name

date
echo "MODIS download for GROUP_ID=${GROUP_ID}, DATES ${START_DATE}-> ${END_DATE}"

# Run the Python script for this group/date range
python3 /path/to/downlaod.py \
    ${GROUP_ID} ${START_DATE} ${END_DATE}


