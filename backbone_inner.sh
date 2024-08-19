cd ./GLPFT

LLAMA_PATH="meta-llama/Llama-2-7b-hf" # your path for initial LLM checkpoint
PORT=12345
BSZ=1
GA=8

EXP_NAME=/toolbench/backbone_trained  # path to save model
export PYTHONPATH=./
python train_lora.py \
    --model_name_or_path $LLAMA_PATH  \
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
    --bf16 \
    --logging_steps 2 \
    --model_max_length 4096 \
    --report_to none \
    --lazy_preprocess True
    # --deepspeed ds_configs/stage3-a100.json \
