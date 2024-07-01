import os
import subprocess
import json
import sys

def main(API_NAME, MODEL='caller', JOB='job', ALL_APIS=False):
    os.chdir('GLPFT')

    # Parameters
    FILENAME = f"../inner_scripts/{JOB}_api"
    CERTAINTY = 'all' if ALL_APIS else 'certain'

    # Default settings
    USE_LORA = True
    BSZ = 4
    GA = 8 // BSZ  # Ensure BSZ * GA = 8
    CONTEXT_LENGTH = 4096

    CATEGORY = API_NAME[0].isupper()
    API_NAME = API_NAME.lower()

    if CATEGORY:
        DATA_PATH = f"dataset/toolbench/new_data/{CERTAINTY}/category/{API_NAME}_train.json"
    elif "_for_" in API_NAME:
        DATA_PATH = f"dataset/toolbench/new_data/{CERTAINTY}/endpoint/{API_NAME}_train.json"
    else:
        DATA_PATH = f"dataset/toolbench/new_data/{CERTAINTY}/api_family/{API_NAME}_train.json"

    EXP_NAME = f"output_patches/{API_NAME}/"

    if MODEL == 'dev':
        INPUT_MODEL = "EleutherAI/pythia-160m"
        CONTEXT_LENGTH = 2048
        TARGET_MODULES = "query_key_value"
        FILENAME = f"../inner_scripts/{API_NAME}_dev_api"
        EXP_NAME = f"output_patches/{API_NAME}_dev/"
    elif MODEL == 'backbone':
        INPUT_MODEL = "saved_models/backbone"
        EXP_NAME = f"output_patches/{API_NAME}_backbone/"
        FILENAME = f"../inner_scripts/{API_NAME}_backbone_api"
    elif MODEL == 'llama':
        INPUT_MODEL = "meta-llama/Llama-2-7b-hf"
        FILENAME = f"../inner_scripts/{API_NAME}_llama_api"
        EXP_NAME = f"output_patches/{API_NAME}_llama/"
    else:
        INPUT_MODEL = "shenwzh3/alpha-umi-caller-7b"
        EXP_NAME = f"output_patches/{API_NAME}/"
        FILENAME = f"../inner_scripts/{API_NAME}_api"

    if CERTAINTY == 'all':
        EXP_NAME = EXP_NAME.rstrip('/') + '_all/'
        FILENAME += '_all'

    print("=====================")
    print(f"API_NAME: {API_NAME}")
    print(f"MODEL: {MODEL}")
    print(f"INPUT_MODEL: {INPUT_MODEL}")
    print(f"FILENAME: {FILENAME}")
    print(f"EXP_NAME: {EXP_NAME}")
    print(f"DATA_PATH: {DATA_PATH}")
    print(f"CONTEXT_LENGTH: {CONTEXT_LENGTH}")
    print(f"BSZ: {BSZ}")
    print(f"GA: {GA}")
    print(f"USE_LORA: {USE_LORA}")
    print("=====================")

    try:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
        NUM_SAMPLES = len(data)
    except FileNotFoundError:
        print(f"File not found: {DATA_PATH}")
        sys.exit(1)

    MIN_SAMPLES = 10000
    EPOCHS = (MIN_SAMPLES + NUM_SAMPLES - 1) // NUM_SAMPLES

    os.chdir('..')

    script_content = f"""#!/bin/bash

cd GLPFT

python train_mine.py \\
    --model_name_or_path {INPUT_MODEL} \\
    --data_path {DATA_PATH} \\
    --output_dir {EXP_NAME} \\
    --num_train_epochs {EPOCHS} \\
    --per_device_train_batch_size {BSZ} \\
    --per_device_eval_batch_size {BSZ} \\
    --gradient_accumulation_steps {GA} \\
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
    --model_max_length {CONTEXT_LENGTH} \\
    --report_to none \\
    --lazy_preprocess False \\
    --bf16 True \\
    --lora {USE_LORA} \\
"""

    if USE_LORA and 'TARGET_MODULES' in locals():
        script_content += f"    --lora_target_modules {TARGET_MODULES} \\\n"

    script_content += '\n'

    with open(f"{FILENAME}.sh", 'w') as f:
        f.write(script_content)

    os.chmod(f"{FILENAME}.sh", 0o755)

    SAMPLE_COUNT = NUM_SAMPLES * EPOCHS

    print(f"Script {FILENAME}.sh has been created and made executable. It is configured to train the model '{MODEL}' using data for the API '{API_NAME}'. {SAMPLE_COUNT} training samples will be seen during training.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Translate Bash script to Python.')
    parser.add_argument('API_NAME', type=str, help='Name of the API')
    parser.add_argument('MODEL', type=str, nargs='?', default='caller', help='Model to use')
    parser.add_argument('JOB', type=str, nargs='?', default='job', help='Job filename')
    parser.add_argument('ALL_APIS', type=bool, nargs='?', default=False, help='Use all APIs or certain APIs')

    args = parser.parse_args()

    main(args.API_NAME, args.MODEL, args.JOB, args.ALL_APIS)
