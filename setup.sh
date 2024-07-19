#!/bin/bash

set -e

create_env() {
  local env_file=$1
  local env_name=$2

  echo "Creating the environment for $env_name"
  if ! conda env create -f "$env_file" -y; then
    echo "Error: Failed to create the conda environment from $env_file"
  else
    echo "Successfully created the environment for $env_name"
  fi
}

activate_env() {
  local env_name=$1

  if ! conda activate "$env_name"; then
    echo "Error: Failed to activate the conda environment $env_name"
  else
    echo "Successfully activated the conda environment $env_name"
  fi
}

download_data() {
  local url=$1
  local output=$2

  echo "Downloading data"
  if ! wget -O "$output" "$url"; then
    echo "Error: Failed to download data"
  else
    echo "Successfully downloaded data"
  fi
}

unzip_data() {
  local zip_file=$1
  local output_dir=$2

  echo "Unzipping data"
  if ! unzip "$zip_file" -d "$output_dir"; then
    echo "Error: Failed to unzip $zip_file"
  else
    echo "Successfully unzipped data"
    rm "$zip_file"
  fi
}

prepare_data_directories() {
  echo "Preparing data directories"

  if [ -d "data" ]; then
    mv data/* .
  else
    echo "Error: 'data' directory does not exist"
  fi

  mkdir -p GLPFT/dataset/toolbench/train/raw_data
}

run_python_scripts() {
  ORI_DATA_DIR="../data"
  RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
  TRAIN_DATA_OUT_DIR="dataset/toolbench/new_data/"
  export PYTHONPATH=./

  echo "Running prepro_raw_stage_1.py"
  if ! python process_data/toolbench/prepro_raw_stage_1.py --data_dir "$ORI_DATA_DIR" --output_path "$RAW_DATA_OUT_DIR"; then
    echo "Error: prepro_raw_stage_1.py script failed"
  else
    echo "Successfully ran prepro_raw_stage_1.py"
  fi

  echo "Running prepro_raw_stage_2.py"
  if ! python process_data/toolbench/prepro_raw_stage_2.py --input_path "$RAW_DATA_OUT_DIR/raw_data_stage_1.json" --output_path "$RAW_DATA_OUT_DIR"; then
    echo "Error: prepro_raw_stage_2.py script failed"
  else
    echo "Successfully ran prepro_raw_stage_2.py"
  fi

  echo "Running prepro_caller.py"
  if ! python process_data/toolbench/prepro_caller.py --input_path "$RAW_DATA_OUT_DIR/raw_data_stage_2.json" --output_path "$TRAIN_DATA_OUT_DIR" --prompt_type toolbench_caller; then
    echo "Error: prepro_caller.py script failed"
  else
    echo "Successfully ran prepro_caller.py"
  fi
}

install_git_lfs() {
  echo "Installing Git LFS"

  if ! git lfs install; then
    echo "Error: Git LFS installation failed"
  else
    echo "Successfully installed Git LFS"
  fi
}

clone_models() {
  echo "Cloning models"

  mkdir -p saved_models/toolbench
  cd saved_models/toolbench || { echo "Error: Directory 'saved_models/toolbench' does not exist"; return; }

  if ! git clone https://www.modelscope.cn/iic/alpha-umi-backbone-7b.git; then
    echo "Error: Failed to clone alpha-umi-backbone-7b repository"
  else
    mv alpha-umi-backbone-7b backbone
    echo "Successfully cloned and moved alpha-umi-backbone-7b"
  fi

  if ! git clone https://huggingface.co/shenwzh3/alpha-umi-caller-7b; then
    echo "Error: Failed to clone alpha-umi-caller-7b repository"
  else
    mv alpha-umi-caller-7b caller
    echo "Successfully cloned and moved alpha-umi-caller-7b"
  fi

  cd ../..
}

# Main script execution
create_env train_env.yml "training"
create_env inference_env.yml "inferencing"
activate_env api_expert_train

cd data || { echo "Error: Directory 'data' does not exist"; return; }

download_data "https://drive.usercontent.google.com/download?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&export=download&authuser=0&confirm=t&uuid=24833ee4-2344-4f5e-a7f4-217dda5807dc&at=APZUnTWRvrQMRFwc--g-NNsSYEy3%3A1717861593828" "data.zip"
unzip_data "data.zip" "data"
prepare_data_directories

cd GLPFT/dataset/toolbench || { echo "Error: Directory 'GLPFT/dataset/toolbench' does not exist"; return; }

run_python_scripts
install_git_lfs
clone_models

echo "Setup completed successfully"
