cd ./GLPFT

RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
TRAIN_DATA_OUT_DIR="dataset/toolbench/train_separated/"
export PYTHONPATH=./

for MODE in 'caller'
do
    python process_data/toolbench/prepro_$MODE.py \
        --input_path $RAW_DATA_OUT_DIR/raw_data_stage_2.json \
        --output_path $TRAIN_DATA_OUT_DIR \
        --prompt_type toolbench_$MODE
done