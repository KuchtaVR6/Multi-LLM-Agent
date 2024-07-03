cd ./GLPFT

export PYTHONPATH=./
python inference_utils/toolbench/infer_pipeline_patches.py \
  --per_device_eval_batch_size 4 \
  --data_path dataset/toolbench/test/in_domain.json \
  --bf16_full_eval \
  --planner_prompt_type toolbench_planner \
  --caller_prompt_type toolbench_caller \
  --summarizer_prompt_type toolbench_summarizer \
  --max_input_length 3750 \
  --output_dir output_patches/test \
  --regular_test_set True \
  --test_backoff True \
  --do_specific_tests True \
  --do_specific_tests_backoff True \
  --specific_test_sets 'certain'
