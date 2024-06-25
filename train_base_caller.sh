cd ./GLPFT

BSZ=6
GA=1

BB_PATH="saved_models/backbone"
DATA_VARIANT=${1:-certain}

EXP_NAME=/toolbench/caller_base_$DATA_VARIANT
export PYTHONPATH=./
python train_mine.py \
    --model_name_or_path $BB_PATH  \
    --data_variant $DATA_VARIANT \
    --data_path dataset/toolbench/new_data/$DATA_VARIANT/train.json \
    --output_dir saved_models/$EXP_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BSZ \
    --per_device_eval_batch_size $BSZ \
    --gradient_accumulation_steps $GA \
    --evaluation_strategy "no" \
    --eval_steps 0 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.2 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True \
    --lora False
