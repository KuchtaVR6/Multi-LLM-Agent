cd ./GLPFT

RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
TRAIN_DATA_OUT_DIR="dataset/toolbench/train_mine"
export PYTHONPATH=./


#python process_data/toolbench/prepro_raw_stage_1.py \
# --data_dir $ORI_DATA_DIR \
# --output_path $RAW_DATA_OUT_DIR
#
#
#python process_data/toolbench/prepro_raw_stage_2.py \
# --input_path $RAW_DATA_OUT_DIR/raw_data_stage_1.json \
# --output_path $RAW_DATA_OUT_DIR

for MODE in 'caller'
do
    python process_data/toolbench/prepro_$MODE.py \
        --input_path $RAW_DATA_OUT_DIR/raw_data_stage_2.json \
        --output_path $TRAIN_DATA_OUT_DIR/train_$MODE.json \
        --prompt_type toolbench_$MODE
done