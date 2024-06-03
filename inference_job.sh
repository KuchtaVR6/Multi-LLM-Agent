#!/bin/bash

export NNODE=1 # num of GPUs

# Grid Engine options
#$ -N inference  # Name of the job
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
conda activate umi

cd ./GLPFT

PORT=12345

PLAN_PATH="shenwzh3/alpha-umi-planner-7b"
CAL_PATH="shenwzh3/alpha-umi-caller-7b"
SUM_PATH="shenwzh3/alpha-umi-summarizer-7b"


LAB_DIR=output_res/toolbench
P_TYPE_PLAN=toolbench_planner
P_TYPE_CAL=toolbench_caller
P_TYPE_SUM=toolbench_summarizer


for DOMAIN in 'in_domain' 'out_of_domain'
do
    export PYTHONPATH=./
    torchrun --nproc_per_node=$NNODE --master_port=$PORT inference_utils/toolbench/infer_pipeline.py \
        --planner_model_name_or_path $PLAN_PATH  \
        --planner_use_lora False \
        --caller_model_name_or_path $CAL_PATH  \
        --caller_use_lora False \
        --summarizer_model_name_or_path $SUM_PATH  \
        --summarizer_use_lora False \
        --per_device_eval_batch_size 4 \
        --data_path dataset/toolbench/test/$DOMAIN.json \
        --bf16_full_eval \
        --planner_prompt_type $P_TYPE_PLAN \
        --caller_prompt_type $P_TYPE_CAL \
	--summarizer_prompt_type $P_TYPE_SUM \
        --max_input_length 3750 \
        --output_dir $LAB_DIR/$DOMAIN \
	--num_infer_samples 10

    python inference_utils/toolbench/evaluate-multi_agent.py \
    --input_path $LAB_DIR/$DOMAIN/predictions.json \
    --output_path $LAB_DIR/$DOMAIN/metrics.json

done

# Deactivate the conda environment
conda deactivate
