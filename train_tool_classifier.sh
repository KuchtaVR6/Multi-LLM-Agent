cd ./GLPFT

export PYTHONPATH=./
python train_categorizer.py \
  --data_path dataset/toolbench/new_data/all/category_train.json \
  --batch_size 1 \
  --num_epochs 1 \
  --model_save_path saved_models/categoriser