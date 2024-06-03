#!/bin/bash

# 1st arg - job name and inner script subname
# 2nd arg - number of gpus

# Define the job script file name based on the first argument
job_script="${1}_job.sh"
inner_script="${1}_inner.sh"

# clear file contents
> "$job_script"

# Create the job script file
echo "#!/bin/bash" >> "$job_script"
echo "" >> "$job_script"

# Append the script contents to the job script file
echo "export NNODE=${2} # num of GPUs" >> "$job_script"
echo "" >> "$job_script"

echo "# Grid Engine options" >> "$job_script"
echo "#\$ -N ${1}  # Name of the job" >> "$job_script"
echo "#\$ -cwd           # Run the job from the current working directory" >> "$job_script"
echo "#\$ -l h_rt=10:00:00  # Request a runtime" >> "$job_script"
echo "#\$ -q gpu          # Submit the job to the gpu queue" >> "$job_script"
echo "#\$ -pe gpu-a100 ${2}  # Request NNODE A100 GPUs" >> "$job_script"
echo "#\$ -l h_vmem=80G    # Request memory per core" >> "$job_script"
echo "" >> "$job_script"

echo "# Load the module system" >> "$job_script"
echo ". /etc/profile.d/modules.sh" >> "$job_script"
echo "" >> "$job_script"

echo "# Load the CUDA module" >> "$job_script"
echo "module load cuda" >> "$job_script"
echo "" >> "$job_script"

echo "# point hugging_face to the right place" >> "$job_script"
echo "export TRANSFORMERS_CACHE=/exports/eddie/scratch/s2595201/hugging_face_cache" >> "$job_script"
echo "" >> "$job_script"

echo "# Activate the conda environment for CUDA" >> "$job_script"
echo "source ../miniconda3/bin/activate base" >> "$job_script"

echo "conda activate umi" >> "$job_script"
echo "" >> "$job_script"

cat "$inner_script" >> "$job_script"
echo "" >> "$job_script"

echo "# Deactivate the conda environment" >> "$job_script"
echo "conda deactivate" >> "$job_script"

