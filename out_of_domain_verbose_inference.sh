cd ./GLPFT

PORT=12345

PLAN_PATH="shenwzh3/alpha-umi-planner-7b"
CAL_PATH="shenwzh3/alpha-umi-caller-7b"
SUM_PATH="shenwzh3/alpha-umi-summarizer-7b"


LAB_DIR=ood_output_res_verbose/toolbench
P_TYPE_PLAN=toolbench_planner
P_TYPE_CAL=toolbench_caller
P_TYPE_SUM=toolbench_summarizer

export PYTHONPATH=./
python inference_utils/toolbench/infer_pipeline.py \
    --planner_model_name_or_path $PLAN_PATH  \
    --planner_use_lora False \
    --caller_model_name_or_path $CAL_PATH  \
    --caller_use_lora False \
    --summarizer_model_name_or_path $SUM_PATH  \
    --summarizer_use_lora False \
    --per_device_eval_batch_size 4 \
    --data_path dataset/toolbench/test/out_of_domain.json \
    --bf16_full_eval \
    --planner_prompt_type $P_TYPE_PLAN \
    --caller_prompt_type $P_TYPE_CAL \
    --summarizer_prompt_type $P_TYPE_SUM \
    --max_input_length 3750 \
    --output_dir $LAB_DIR \
    --num_infer_samples 2 \
    --verbose_outputs True
