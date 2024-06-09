cd ./GLPFT

ORI_DATA_DIR="../data" # your data path to save the toolbench raw data
RAW_DATA_OUT_DIR="dataset/toolbench/train/raw_data"
export PYTHONPATH=./


python process_data/toolbench/prepro_raw_stage_1.py \
 --data_dir $ORI_DATA_DIR \
 --output_path $RAW_DATA_OUT_DIR


python process_data/toolbench/prepro_raw_stage_2.py \
 --input_path $RAW_DATA_OUT_DIR/raw_data_stage_1.json \
 --output_path $RAW_DATA_OUT_DIR