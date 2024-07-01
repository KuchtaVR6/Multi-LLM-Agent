#!/usr/bin/env python3

import os
import json
import sys


def create_script_content(input_model, data_path, exp_name, epochs, bsz, ga, context_length, use_lora, filename,
                          target_modules=None):
    script_content = f"""#!/bin/bash

cd GLPFT

python train_mine.py \\
    --model_name_or_path {input_model} \\
    --data_path {data_path} \\
    --output_dir {exp_name} \\
    --num_train_epochs {epochs} \\
    --per_device_train_batch_size {bsz} \\
    --per_device_eval_batch_size {bsz} \\
    --gradient_accumulation_steps {ga} \\
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
    --model_max_length {context_length} \\
    --report_to none \\
    --lazy_preprocess False \\
    --bf16 True \\
    --lora {use_lora} \\
"""

    if use_lora and target_modules:
        script_content += f"    --lora_target_modules {target_modules} \\\n"

    script_content += '\n'

    with open(f"{filename}.sh", 'w') as f:
        f.write(script_content)

    os.chmod(f"{filename}.sh", 0o755)


def main(api_name, model='caller', all_apis=False):
    os.chdir('GLPFT')

    certainty = 'all' if all_apis else 'certain'

    # Default settings
    use_lora = True
    bsz = 4
    ga = 8 // bsz  # Ensure BSZ * GA = 8
    context_length = 4096

    category = api_name[0].isupper()
    api_name = api_name.lower()

    if category:
        data_path = f"dataset/toolbench/new_data/{certainty}/category/{api_name}_train.json"
    elif "_for_" in api_name:
        data_path = f"dataset/toolbench/new_data/{certainty}/endpoint/{api_name}_train.json"
    else:
        data_path = f"dataset/toolbench/new_data/{certainty}/api_family/{api_name}_train.json"

    exp_name = f"output_patches/{api_name}/"

    model_settings = {
        'dev': ("EleutherAI/pythia-160m", 2048, "query_key_value", f"../inner_scripts/{api_name}_dev_api",
                f"output_patches/{api_name}_dev/"),
        'backbone': ("saved_models/backbone", 4096, None, f"../inner_scripts/{api_name}_backbone_api",
                     f"output_patches/{api_name}_backbone/"),
        'llama': ("meta-llama/Llama-2-7b-hf", 4096, None, f"../inner_scripts/{api_name}_llama_api",
                  f"output_patches/{api_name}_llama/"),
        "caller[vimeo]": (
        "placeholder_model", "placeholder_context_length", "placeholder_target_modules", "placeholder_filename",
        "placeholder_experiment_name")
    }

    if model in model_settings:
        input_model, context_length, target_modules, filename, exp_name = model_settings[model]
    else:
        input_model = "shenwzh3/alpha-umi-caller-7b"
        filename = f"../inner_scripts/{api_name}_api"

    if certainty == 'all':
        exp_name = exp_name.rstrip('/') + '_all/'
        filename += '_all'

    print("=====================")
    print(f"API_NAME: {api_name}")
    print(f"MODEL: {model}")
    print(f"INPUT_MODEL: {input_model}")
    print(f"FILENAME: {filename}")
    print(f"EXP_NAME: {exp_name}")
    print(f"DATA_PATH: {data_path}")
    print(f"CONTEXT_LENGTH: {context_length}")
    print(f"BSZ: {bsz}")
    print(f"GA: {ga}")
    print(f"USE_LORA: {use_lora}")
    print("=====================")

    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        num_samples = len(data)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        sys.exit(1)

    min_samples = 10000
    epochs = (min_samples + num_samples - 1) // num_samples

    os.chdir('..')

    create_script_content(input_model, data_path, exp_name, epochs, bsz, ga, context_length, use_lora, filename,
                          locals().get('target_modules'))

    sample_count = num_samples * epochs

    print(
        f"Script {filename}.sh has been created and made executable. It is configured to train the model '{model}' using data for the API '{api_name}'. {sample_count} training samples will be seen during training.")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Translate Bash script to Python.')
    parser.add_argument('api_name', type=str, help='Name of the API')
    parser.add_argument('model', type=str, nargs='?', default='caller', help='Model to use')
    parser.add_argument('all_apis', type=bool, nargs='?', default=False, help='Use all APIs or certain APIs')

    args = parser.parse_args()

    main(args.api_name, args.model, args.all_apis)
