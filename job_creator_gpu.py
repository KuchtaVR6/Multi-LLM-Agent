#!/usr/bin/env python3

import os
import sys
import argparse


def jobify(job_path, use_inference=False):
    number_of_gpus = 1

    small_job_name = job_path.rsplit('/', -1)[1]

    job_script = f"../jobs/{small_job_name}_job.sh"
    inner_script = f"{job_path}.sh"

    # Check if the inner script exists
    if not os.path.isfile(inner_script):
        print(f"File {inner_script} does not exist.")
        sys.exit(1)

    # Clear file contents if it exists
    open(job_script, 'w').close()

    # Create the job script file
    with open(job_script, 'w') as f:
        f.write(f"""#!/bin/bash

# Grid Engine options
#$ -N {small_job_name}_new  # Name of the job
#$ -wd /exports/eddie/scratch/s2595201/jobs/logs # Run the job from the scratch directory
#$ -l h_rt=12:00:00  # Request a runtime
#$ -q gpu          # Submit the job to the gpu queue
#$ -pe gpu-a100 {number_of_gpus}  # Request NNODE A100 GPUs
#$ -l h_vmem=80G    # Request memory per core
#$ -l rl9=true     # Use rocky linux true

# Load the module system
. /etc/profile.d/modules.sh

# Load the CUDA module
module load cuda

# Activate the conda environment for CUDA
source /exports/csce/eddie/inf/groups/dawg/miniconda3/bin/activate base

cd /exports/eddie/scratch/s2595201/Multi-LLM-Agent

conda activate api_expert{"_inference" if use_inference else "_train"}

export TOKENIZERS_PARALLELIZM=false

export HF_HOME=/exports/csce/eddie/inf/groups/dawg/HuggingFaceCache

export TRANSFORMERS_CACHE=/exports/csce/eddie/inf/groups/dawg/HuggingFaceCache

huggingface-cli login --token $HF_TOKEN --add-to-git-credential

""")
        # Append the inner script contents
        with open(inner_script, 'r') as inner_f:
            f.write(inner_f.read())

        f.write("\nconda deactivate\n")

    # Make the generated script executable
    os.chmod(job_script, 0o755)

    print(f"Script {job_script} has been created and made executable.")

    return job_script


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate job script for a given job.')
    parser.add_argument('job_name', type=str, help='Name of the job')
    parser.add_argument('--inference', action='store_true', help='Use the _inference conda environment instead of _train')

    args = parser.parse_args()

    jobify(args.job_name, args.inference)
