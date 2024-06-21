#!/bin/bash

# 1st arg - job name and inner script name
# 2nd arg - number of gpus

# Define the job script file name based on the first argument
job_script="../jobs/${1}_job.sh"
inner_script="${1}.sh"

# Check if the file exists
if [ ! -f "$inner_script" ]; then
  echo "File $inner_script does not exist."
  exit 1
fi

# Clear file contents
> "$job_script"

# Create the job script file
cat <<EOF > "$job_script"
#!/bin/bash

# Grid Engine options
#$ -N ${1}  # Name of the job
#$ -wd /exports/eddie/scratch/s2595201/Multi-LLM-Agent # Run the job from the scratch directory
#$ -l h_rt=6:00:00  # Request a runtime
#$ -q gpu          # Submit the job to the gpu queue
#$ -pe gpu-a100 1  # Request NNODE A100 GPUs
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

EOF

# Append the inner script contents
cat "$inner_script" >> "$job_script"
echo "" >> "$job_script"

# Deactivate the conda environment
echo "conda deactivate" >> "$job_script"

# Make the generated script executable
chmod +x "$job_script"

echo "Script ${job_script} has been created and made executable."