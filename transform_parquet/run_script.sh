#!/bin/bash

#SBATCH --nodes=1
#SBATCH --mem-per-cpu=10GB
#SBATCH --time=01:00:00

module load stack/2024-06 python/3.11.6
source /cluster/home/anmusso/quantization/myenv/bin/activate

python run_script.py "$@"