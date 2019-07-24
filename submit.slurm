#!/bin/bash

#SBATCH --time=99:00:00
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%J.out

# Give this process 1 task (per GPU, but only one GPU), then assign eight 8per task
# (so 8 cores overall).  Then enforce that slurm assigns only CPUs that are on the
# socket closest to the GPU you get.

# If you want two GPUs:
# #SBATCH --ntasks=2
# #SBATCH --gres=gpu:2


# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

## Load the TensorFlow module
#module load anaconda/anaconda3-5.3.0
#
## List the modules that are loaded
#module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m

echo

## Activate the GPU version of TensorFlow
#source activate tensorflow-gpu

# OR, instead:  Activate the non-GPU version of TensorFlow
#source activate tensorflow

echo "${1}"

# Run TensorFlow
echo
${1}
echo

# You're done!
echo "Ending script..."
date

