#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=80:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=5G
#SBATCH --tmp=4000                        # per node!!

#SBATCH --job-name=power_unit
#SBATCH --output=./euler/degree.out  # to be modified
#SBATCH --error=./euler/degree.err   # to be modified

# run experiment
python ./connectivity.py