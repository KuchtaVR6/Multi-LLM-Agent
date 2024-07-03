#!/usr/bin/env python3

import os
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: script.py <job_name>")
        sys.exit(1)

    job_name = sys.argv[1]
    number_of_gpus = 1

    job_script = f"../jobs/{job_name}_job.sh"
    inner_script = f"../inner_scripts/{job_name}.sh"

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
#$ -N {job_name}_new  # Name of the job
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

conda activate api_expert_env

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


if __name__ == "__main__":
    main()
