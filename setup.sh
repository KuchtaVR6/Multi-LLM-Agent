echo "Creating the environment"

conda env create -f api_expert_env.yml -y
conda activate api_expert_env

cd data || exit

echo "Downloading data"
wget "https://drive.usercontent.google.com/download?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&export=download&authuser=0&confirm=t&uuid=24833ee4-2344-4f5e-a7f4-217dda5807dc&at=APZUnTWRvrQMRFwc--g-NNsSYEy3%3A1717861593828"
mv 'download?id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&export=download&authuser=0&confirm=t&uuid=24833ee4-2344-4f5e-a7f4-217dda5807dc&at=APZUnTWRvrQMRFwc--g-NNsSYEy3:1717861593828' data.zip

echo "Unzipping data"
unzip data.zip
rm data.zip
mv data/* .
rmdir data

echo "Data unzipped data"
cd ../GLPFT/dataset/toolbench || exit
mkdir -p train
cd train || exit
mkdir -p raw_data

cd ../../..

echo "Current Working Directory: $(pwd)"

ORI_DATA_DIR="../data" # your data path to save the toolbench raw data
RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
TRAIN_DATA_OUT_DIR="dataset/toolbench/new_data/"
export PYTHONPATH=./


python process_data/toolbench/prepro_raw_stage_1.py \
 --data_dir $ORI_DATA_DIR \
 --output_path $RAW_DATA_OUT_DIR

python process_data/toolbench/prepro_raw_stage_2.py \
 --input_path $RAW_DATA_OUT_DIR/raw_data_stage_1.json \
 --output_path $RAW_DATA_OUT_DIR

python process_data/toolbench/prepro_caller.py \
    --input_path $RAW_DATA_OUT_DIR/raw_data_stage_2.json \
    --output_path $TRAIN_DATA_OUT_DIR \
    --prompt_type toolbench_caller

echo ">>>>>>>>>>>>>> Dataset ready"
echo ">>>>>>>>>>>>>> Installing base models"

git lfs install
mkdir -p saved_models
cd saved_models || exit
mkdir -p toolbench
cd toolbench || exit

echo "Current Working Directory: $(pwd)"

git clone https://www.modelscope.cn/iic/alpha-umi-backbone-7b.git
git clone https://huggingface.co/shenwzh3/alpha-umi-caller-7b

mv alpha-umi-backbone-7b backbone
mv alpha-umi-caller-7b caller

echo ">>>>>>>>>>>>>> Repository ready"


