#!/bin/bash

export NNODE=1 # num of GPUs

# Grid Engine options
#$ -N inference_test  # Name of the job
#$ -cwd           # Run the job from the current working directory
#$ -l h_rt=5:00:00  # Request a runtime
#$ -q gpu          # Submit the job to the gpu queue
#$ -pe gpu-a100 1  # Request NNODE A100 GPUs
#$ -l h_vmem=80G    # Request memory per core

# Load the module system
. /etc/profile.d/modules.sh

# Load the CUDA module
module load cuda

# point hugging_face to the right placex
export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2595201/hugging_face_cache

# Activate the conda environment for CUDA
source ../miniconda3/bin/activate base
conda activate umi

echo $CUDA_VISIBLE_DEVICES

. inference_inner.sh > inner.log

# Deactivate the conda environment
conda deactivate
