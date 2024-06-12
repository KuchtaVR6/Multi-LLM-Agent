#!/bin/bash

cd GLPFT

# Parameters
API_NAME=$1
MODEL=$2
FILENAME="../inner_scripts/${3:-job}_api"  # Default value of FILENAME is 'job' if not provided

# Default settings
USE_LORA=true
BSZ=4  # Default value of BSZ is 4
GA=$((8 / BSZ))   # Ensure BSZ * GA = 8

CONTEXT_LENGTH=4096

if [[ $API_NAME == *"_for_"* ]]; then
  DATA_PATH="dataset/toolbench/train_separated/certain/${API_NAME}.json"
else
  DATA_PATH="dataset/toolbench/train_per_api/${API_NAME}.json"
fi


DATA_PATH="dataset/toolbench/train_separated/certain/${API_NAME}.json"
EXP_NAME=output_pathes/${API_NAME}/

# Determine the input model based on the third parameter
if [[ $MODEL == "dev" ]]; then
    INPUT_MODEL="EleutherAI/pythia-160m"
    CONTEXT_LENGTH=2048
    TARGET_MODULES="query_key_value"
elif [[ $MODEL == "backbone" ]]; then
    INPUT_MODEL="saved_models/backbone"
    EXP_NAME=output_pathes/${API_NAME}_backbone/
elif [[ $MODEL == "caller" ]]; then
    INPUT_MODEL="shenwzh3/alpha-umi-caller-7b"
else
    echo "Invalid model type. Please choose from {dev, backbone, caller}."
    exit 1
fi

NUM_SAMPLES=$(python3 -c "
import json
with open('${DATA_PATH}', 'r') as f:
    data = json.load(f)
print(len(data))
")
MIN_SAMPLES=10000
EPOCHS=$(( (MIN_SAMPLES + NUM_SAMPLES - 1) / NUM_SAMPLES ))

cd ..

# Create the output script
cat << EOF > ${FILENAME}.sh
#!/bin/bash

cd GLPFT

python train_mine.py \\
    --model_name_or_path ${INPUT_MODEL} \\
    --data_path ${DATA_PATH} \\
    --output_dir ${EXP_NAME} \\
    --num_train_epochs ${EPOCHS} \\
    --per_device_train_batch_size ${BSZ} \\
    --per_device_eval_batch_size ${BSZ} \\
    --gradient_accumulation_steps ${GA} \\
    --eval_strategy "no" \\
    --eval_steps 0 \\
    --save_strategy "steps" \\
    --save_steps 500 \\
    --save_total_limit 8 \\
    --learning_rate 1e-5 \\
    --warmup_ratio 0.2 \\
    --lr_scheduler_type "cosine" \\
    --gradient_checkpointing True \\
    --logging_steps 2 \\
    --model_max_length ${CONTEXT_LENGTH} \\
    --report_to none \\
    --lazy_preprocess False \\
    --bf16 True \\
    --lora ${USE_LORA} \\
EOF

# Conditionally add the line if TARGET_MODULES is set
if [[ -n $TARGET_MODULES ]]; then
  if [[ $USE_LORA == true ]]; then
    echo "    --lora_target_modules ${TARGET_MODULES} \\" >> ${FILENAME}.sh
  fi
fi

# Make the generated script executable
chmod +x ${FILENAME}.sh

SAMPLE_COUNT=$((NUM_SAMPLES * EPOCHS))


echo "Script ${FILENAME}.sh has been created and made executable. It is configured to train the model '${MODEL}' using data for the API '${API_NAME}'. ${SAMPLE_COUNT} training samples will be seen during training."
