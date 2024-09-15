#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=80:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=3G
#SBATCH --tmp=4000                        # per node!!

#SBATCH --job-name=power_unit
#SBATCH --output=./euler/10-100,ring.out  # to be modified
#SBATCH --error=./euler/10-100,ring.err   # to be modified

# run experiment
python ./test.py