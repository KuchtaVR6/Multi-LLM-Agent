#!/bin/bash

# export NNODE=1 # num of GPUs

# Grid Engine options
#$ -N caller  # Name of the job
#$ -cwd           # Run the job from the current working directory
#$ -l h_rt=10:00:00  # Request a runtime
#$ -q gpu          # Submit the job to the gpu queue
#$ -pe gpu-a100 1  # Request NNODE A100 GPUs
#$ -l h_vmem=80G    # Request memory per core

# Load the module system
. /etc/profile.d/modules.sh

# Load the CUDA module
module load cuda

# point hugging_face to the right place
export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2595201/hugging_face_cache

# Activate the conda environment for CUDA
source ../miniconda3/bin/activate base
conda activate umi_vanilla

cd ./GLPFT

BB_PATH="saved_models/toolbench/backbone" # your path for initial LLM checkpoint
PORT=12345
BSZ=8
GA=1

EXP_NAME=/toolbench/caller  # path to save model
export PYTHONPATH=./
python train_lora.py \
    --model_name_or_path $BB_PATH  \
    --data_path dataset/toolbench/train/train_planner.json\
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True
    # --deepspeed ds_configs/stage3-a100.json \


# Deactivate the conda environment
conda deactivate
