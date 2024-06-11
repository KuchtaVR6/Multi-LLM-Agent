LAB_DIR=output_res/toolbench

cd GLPFT

for DOMAIN in 'in_domain'
do
  python inference_utils/toolbench/evaluate-multi_agent.py \
        --input_path $LAB_DIR/$DOMAIN/predictions.json \
        --output_path $LAB_DIR/$DOMAIN/metrics.json
done