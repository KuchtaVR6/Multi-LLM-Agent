#!/usr/bin/env python3

from job_creator_gpu import jobify
import argparse
import os


def create_inner_script(model_suffix):
    inner_script_path = f"../inner_scripts/infer_{model_suffix}.sh"

    inner_script_content = f"""#!/bin/bash

cd ./GLPFT

export PYTHONPATH=./
python inference_utils/toolbench/infer_pipeline_patches.py \\
  --per_device_eval_batch_size 4 \\
  --data_path dataset/toolbench/test/in_domain.json \\
  --bf16_full_eval \\
  --planner_prompt_type toolbench_planner \\
  --caller_prompt_type toolalpaca_caller \\
  --summarizer_prompt_type toolbench_summarizer \\
  --max_input_length 3750 \\
  --output_dir output_patches/test \\
  --regular_test_set True \\
  --test_backoff False \\
  --do_specific_tests True \\
  --do_specific_tests_backoff True \\
  --specific_test_sets 'all' \\
  --model_suffix '{model_suffix}'
"""

    with open(inner_script_path, 'w') as f:
        f.write(inner_script_content)

    os.chmod(inner_script_path, 0o755)

    return inner_script_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate job scripts for inference.")
    parser.add_argument("model_suffix", type=str, help="The model suffix to be used in the inference script.")
    args = parser.parse_args()

    model_suffix = args.model_suffix

    # Create the inner script
    inner_script_path = create_inner_script(model_suffix)

    # Generate the job script
    jobify(inner_script_path[:-3], use_inference=True)
