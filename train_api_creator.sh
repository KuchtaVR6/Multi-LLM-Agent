#!/bin/bash

cd GLPFT

# Parameters
API_NAME=$1
MODEL=${2:-caller}  # Default value of MODEL is 'caller' if not provided
FILENAME="../inner_scripts/${3:-job}_api"  # Default value of FILENAME is 'job' if not provided
ALL_APIS=${4:-false}  # Certain or all by default its certain cases

# Default settings
USE_LORA=true
BSZ=4  # Default value of BSZ is 4
GA=$((8 / BSZ))   # Ensure BSZ * GA = 8

CONTEXT_LENGTH=4096

if [[ $ALL_APIS == true ]]; then
  CERTAINTY="all"
else
  CERTAINTY="certain"
fi

# Check if the first letter of API_NAME is uppercase
if [[ $API_NAME =~ ^[A-Z] ]]; then
  CATEGORY=true
else
  CATEGORY=false
fi

# Lowercase the API_NAME
API_NAME=$(echo "$API_NAME" | tr '[:upper:]' '[:lower:]')

# Determine the DATA_PATH based on CATEGORY flag
if [[ $CATEGORY == true ]]; then
  DATA_PATH="dataset/toolbench/new_data/${CERTAINTY}/category/${API_NAME}_train.json"
elif [[ $API_NAME == *"_for_"* ]]; then
  DATA_PATH="dataset/toolbench/new_data/${CERTAINTY}/endpoint/${API_NAME}_train.json"
else
  DATA_PATH="dataset/toolbench/new_data/${CERTAINTY}/api_family/${API_NAME}_train.json"
fi

EXP_NAME="output_pathes/${API_NAME}/"

# Determine the input model based on the second parameter
if [[ $MODEL == "dev" ]]; then
    INPUT_MODEL="EleutherAI/pythia-160m"
    CONTEXT_LENGTH=2048
    TARGET_MODULES="query_key_value"
    FILENAME="../inner_scripts/${API_NAME}_dev_api"
    EXP_NAME="output_patches/${API_NAME}_dev/"
elif [[ $MODEL == "backbone" ]]; then
    INPUT_MODEL="saved_models/backbone"
    EXP_NAME="output_patches/${API_NAME}_backbone/"
    FILENAME="../inner_scripts/${API_NAME}_backbone_api"
elif [[ $MODEL == "llama" ]]; then
    INPUT_MODEL="meta-llama/Llama-2-7b-hf"
    FILENAME="../inner_scripts/${API_NAME}_llama_api"
    EXP_NAME="output_patches/${API_NAME}_llama/"
else
    INPUT_MODEL="shenwzh3/alpha-umi-caller-7b"
    EXP_NAME="output_patches/${API_NAME}/"
    FILENAME="../inner_scripts/${API_NAME}_api"
fi

if [[ $CERTAINTY == "all"  ]]; then
    EXP_NAME+="_all"
    FILENAME+="_all"
fi

echo "====================="
echo "API_NAME: $API_NAME"
echo "MODEL: $MODEL"
echo "INPUT_MODEL: $INPUT_MODEL"
echo "FILENAME: $FILENAME"
echo "EXP_NAME: $EXP_NAME"
echo "DATA_PATH: $DATA_PATH"
echo "CONTEXT_LENGTH: $CONTEXT_LENGTH"
echo "BSZ: $BSZ"
echo "GA: $GA"
echo "USE_LORA: $USE_LORA"
echo "====================="

NUM_SAMPLES=$(python3 -c "
import json
import sys
try:
    with open('${DATA_PATH}', 'r') as f:
        data = json.load(f)
    print(len(data))
except FileNotFoundError:
    sys.exit(1)
")

if [[ $? -ne 0 ]]; then
  echo "File not found: ${DATA_PATH}"
  exit 1
fi

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
