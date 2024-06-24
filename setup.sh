#!/bin/bash

set -e

echo "Creating the environment"

if ! conda env create -f api_expert_env.yml -y; then
  echo "Error: Failed to create the conda environment from api_expert_env.yml"
  exit 1
fi

if ! conda activate api_expert_env; then
  echo "Error: Failed to activate the conda environment"
  exit 1
fi

cd data || { echo "Error: Directory 'data' does not exist"; exit 1; }

echo "Downloading data"
if ! wget -O data.zip "https://drive.usercontent.google.com/download?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&export=download&authuser=0&confirm=t&uuid=24833ee4-2344-4f5e-a7f4-217dda5807dc&at=APZUnTWRvrQMRFwc--g-NNsSYEy3%3A1717861593828"; then
  echo "Error: Failed to download data"
  exit 1
fi

echo "Unzipping data"
if ! unzip data.zip; then
  echo "Error: Failed to unzip data.zip"
  exit 1
fi

rm data.zip
mv data/* . || { echo "Error: Failed to move data"; exit 1; }
rmdir data || { echo "Error: Failed to remove 'data' directory"; exit 1; }

echo "Data unzipped successfully"
cd ../GLPFT/dataset/toolbench || { echo "Error: Directory 'GLPFT/dataset/toolbench' does not exist"; exit 1; }
mkdir -p train
cd train || { echo "Error: Directory 'train' does not exist"; exit 1; }
mkdir -p raw_data

cd ../../.. || { echo "Error: Failed to navigate to the project root"; exit 1; }

echo "Current Working Directory: $(pwd)"

ORI_DATA_DIR="../data" # your data path to save the toolbench raw data
RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
TRAIN_DATA_OUT_DIR="dataset/toolbench/new_data/"
export PYTHONPATH=./

if ! python process_data/toolbench/prepro_raw_stage_1.py --data_dir "$ORI_DATA_DIR" --output_path "$RAW_DATA_OUT_DIR"; then
  echo "Error: prepro_raw_stage_1.py script failed"
  exit 1
fi

if ! python process_data/toolbench/prepro_raw_stage_2.py --input_path "$RAW_DATA_OUT_DIR/raw_data_stage_1.json" --output_path "$RAW_DATA_OUT_DIR"; then
  echo "Error: prepro_raw_stage_2.py script failed"
  exit 1
fi

if ! python process_data/toolbench/prepro_caller.py --input_path "$RAW_DATA_OUT_DIR/raw_data_stage_2.json" --output_path "$TRAIN_DATA_OUT_DIR" --prompt_type toolbench_caller; then
  echo "Error: prepro_caller.py script failed"
  exit 1
fi

echo ">>>>>>>>>>>>>> Dataset ready"
echo ">>>>>>>>>>>>>> Installing base models"

if ! git lfs install; then
  echo "Error: Git LFS installation failed"
  exit 1
fi

mkdir -p saved_models
cd saved_models || { echo "Error: Directory 'saved_models' does not exist"; exit 1; }
mkdir -p toolbench
cd toolbench || { echo "Error: Directory 'toolbench' does not exist"; exit 1; }

echo "Current Working Directory: $(pwd)"

if ! git clone https://www.modelscope.cn/iic/alpha-umi-backbone-7b.git; then
  echo "Error: Failed to clone alpha-umi-backbone-7b repository"
  exit 1
fi

if ! git clone https://huggingface.co/shenwzh3/alpha-umi-caller-7b; then
  echo "Error: Failed to clone alpha-umi-caller-7b repository"
  exit 1
fi

mv alpha-umi-backbone-7b backbone || { echo "Error: Failed to rename alpha-umi-backbone-7b to backbone"; exit 1; }
mv alpha-umi-caller-7b caller || { echo "Error: Failed to rename alpha-umi-caller-7b to caller"; exit 1; }

echo ">>>>>>>>>>>>>> Repository ready"
