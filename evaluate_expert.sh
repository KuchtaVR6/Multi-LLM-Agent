cd GLPFT
export PYTHONPATH=./

for DOMAIN in 'in_domain'
do
  python inference_utils/toolbench/evaluate-expert-improvements.py \
        --input_path_backoff output_verbose_res/predictions_caller.json \
        --input_path_expert output_patches/test/ott_details_caller/predictions.json \
        --id_sample_matching True \
        --output_path output_patches/test/ott_details_caller/metrics.json
done