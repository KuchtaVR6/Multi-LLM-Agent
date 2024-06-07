#!/bin/bash

# Parameters
TARGET=$1
MODEL=$2
USE_LORA=$3
CONTEXT_LENGTH=4096

# Determine the input model based on the second parameter
if [[ $MODEL == "LLAMA" ]]; then
    INPUT_MODEL="meta-llama/Llama-2-7b-hf"
elif [[ $MODEL == "PYTHIA" ]]; then
    INPUT_MODEL="EleutherAI/pythia-160m"
    CONTEXT_LENGTH=2048
else
    INPUT_MODEL="saved_models/toolbench/${TARGET}_trained"
fi

# Determine the script to run based on the third parameter
if [[ $USE_LORA == "true" ]]; then
    SCRIPT="train_lora"
else
    SCRIPT="train_mem"
fi

# Create the output script
cat << EOF > job_inner.sh
#!/bin/bash

cd GLPFT

INPUT_MODEL="${INPUT_MODEL}"
BSZ=8
GA=1

EXP_NAME=/toolbench/${TARGET}_trained
python ${SCRIPT}.py \\
    --model_name_or_path \$INPUT_MODEL \\
    --data_path dataset/toolbench/train/train_${TARGET}.json\\
    --output_dir saved_models/\$EXP_NAME \\
    --num_train_epochs 2 \\
    --per_device_train_batch_size \$BSZ \\
    --per_device_eval_batch_size \$BSZ \\
    --gradient_accumulation_steps \$GA \\
    --evaluation_strategy "no" \\
    --eval_steps 0 \\
    --save_strategy "steps" \\
    --save_steps 500 \\
    --save_total_limit 8 \\
    --learning_rate 5e-5 \\
    --warmup_ratio 0.4 \\
    --lr_scheduler_type "cosine" \\
    --gradient_checkpointing True \\
    --logging_steps 2 \\
    --model_max_length ${CONTEXT_LENGTH} \\
    --report_to none \\
    --lazy_preprocess True
EOF

# Make the generated script executable
chmod +x job_inner.sh

echo "Script job_inner.sh has been created and made executable."
