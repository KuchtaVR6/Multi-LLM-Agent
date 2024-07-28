#!/usr/bin/env python3

import os
import json
import sys

from job_creator_gpu import jobify
from GLPFT.supportedModels import get_model_path_on_suffix

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


def doubly_patched_output(model_base_full, patch_name, is_all=False):
    if is_all:
        model_base_full += '_all'

    return f"output_patches/{patch_name}_{model_base_full}"


def merge_patch_and_save(model_suffix, patch_path, output_dir):
    import transformers
    from peft import PeftConfig, get_peft_model

    if model_suffix is None:
        model_suffix = 'caller'

    model_name_or_path = get_model_path_on_suffix(model_suffix)

    if '_all' in patch_path:
        full_patch_path = patch_path.replace('/', f'/trained_on_all/{model_suffix}/', 1)
    else:
        full_patch_path = patch_path.replace('/', f'/{model_suffix}/', 1)
    current_config = PeftConfig.from_pretrained(full_patch_path)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )

    model = get_peft_model(model, current_config)

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)


def main(api_name, model='caller', all_apis=False):
    os.chdir('GLPFT')

    certainty = 'all' if all_apis else 'certain'

    # Default settings
    use_lora = True
    bsz = 4
    ga = 8 // bsz  # Ensure BSZ * GA = 8
    context_length = 4096

    if api_name[0].isupper():
        expert_target_type = 'category'
    elif "_for_" in api_name:
        expert_target_type = 'endpoint'
    else:
        expert_target_type = 'api_family'

    api_name = api_name.lower()
    data_path = f"dataset/toolbench/new_data/{certainty}/{expert_target_type}/{api_name}_train.json"

    exp_name = f"output_patches/{api_name}/"

    model_settings = {
        'dev': ("EleutherAI/pythia-160m",
                2048,
                "query_key_value",
                f"../inner_scripts/{api_name}_dev_api",
                f"output_patches/{api_name}_dev/"),

        'backbone': ("saved_models/backbone",
                     4096,
                     None,
                     f"../inner_scripts/{api_name}_backbone_api",
                     f"output_patches/{api_name}_backbone/"),

        'llama': ("meta-llama/Llama-2-7b-hf",
                  4096,
                  None,
                  f"../inner_scripts/{api_name}_llama_api",
                  f"output_patches/{api_name}_llama/"),

        'caller': ("shenwzh3/alpha-umi-caller-7b",
                   4096,
                   None,
                   f"../inner_scripts/{api_name}_api",
                   f"output_patches/{api_name}/"),
    }

    for initial_base_model in model_settings:
        if model.startswith(initial_base_model):
            input_model, context_length, target_modules, filename, exp_name = model_settings[initial_base_model]

            if '[' in model and ']' in model:
                previous_target = model.rsplit('[', 1)[1].split(']', 1)[0]
                if '[' in model.rsplit('[', 1)[0]:
                    previous_base = model.rsplit('[', 1)[0] + ']'
                else:
                    previous_base = initial_base_model
                if previous_base == 'caller':
                    path_to_patch = previous_target
                else:
                    path_to_patch = previous_target + '_' + previous_base
                patch_dir = 'output_patches/' + path_to_patch + '/'
                output_dir = 'saved_models/' + model + '/'

                if all_apis:
                    output_dir = output_dir.replace('/', '/trained_on_all/', 1)

                if not os.path.exists(output_dir):
                    print('pre-training merge...')
                    merge_patch_and_save(previous_base, patch_dir, output_dir)

                input_model = output_dir
                filename = filename.replace(f'_{initial_base_model}', '')
                filename = filename.replace('_api', f'_{model}_api')
                exp_name = doubly_patched_output(model, api_name, all_apis)

    if certainty == 'all':
        exp_name = exp_name.rstrip('/') + '_all/'
        # filename += '_all'
        model_folder_out = 'trained_on_all/' + model
    else:
        model_folder_out = model

    exp_name = exp_name.replace('/', f'/{model_folder_out}/', 1)

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

    return filename


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Translate Bash script to Python.')
    parser.add_argument('api_name', type=str, help='Name of the API')
    parser.add_argument('model', type=str, nargs='?', default='caller', help='Model to use')
    parser.add_argument('all_apis', type=bool, nargs='?', default=False, help='Use all APIs or certain APIs')

    args = parser.parse_args()

    filename = main(args.api_name, args.model, args.all_apis)
    job_filename = jobify(filename)
