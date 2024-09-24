#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=120:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=4000                        # per node!!

#SBATCH --job-name=power_unit
#SBATCH --output=./euler/degree-stable-100.out  # to be modified
#SBATCH --error=./euler/degree-stable-100.err   # to be modified

# run experiment
python ./connectivity.py