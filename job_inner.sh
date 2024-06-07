#!/bin/bash

cd GLPFT

INPUT_MODEL="EleutherAI/pythia-160m"
BSZ=1
GA=8

EXP_NAME=/toolbench/backbone_trained
python train_lora.py \
    --model_name_or_path $INPUT_MODEL \
    --data_path dataset/toolbench/train/train_backbone.json\
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
    --logging_steps 2 \
    --model_max_length 2048 \
    --report_to none \
    --lazy_preprocess True
