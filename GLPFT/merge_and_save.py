import transformers
from peft import PeftConfig, get_peft_model
import argparse
import os


def merge_patch_and_save(model_suffix, patch_path, output_dir):
    if model_suffix is None:
        model_suffix = 'caller'

    if model_suffix == 'llama':
        model_name_or_path = "meta-llama/Llama-2-7b-hf"
    elif model_suffix == 'dev':
        model_name_or_path = "EleutherAI/pythia-160m"
    else:
        model_name_or_path = f'saved_models/{model_suffix}'

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        base_model.resize_token_embeddings(len(tokenizer))

    model = PeftConfig.from_pretrained(base_model, patch_path)

    model.merge_adapter()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Merge patch and save the result.')

    parser.add_argument(
        'model_suffix',
        type=str,
        help='The suffix of the model to be merged.'
    )

    parser.add_argument(
        'patch_path',
        type=str,
        help='The path to the patch file.'
    )

    parser.add_argument(
        'output_dir',
        type=str,
        help='The directory where the output should be saved.'
    )

    args = parser.parse_args()

    if not os.path.exists(args.patch_path):
        raise FileNotFoundError(f"The patch file {args.patch_path} does not exist.")

    full_output_dir = 'saved_models/' + args.output_dir

    if not os.path.isdir(full_output_dir):
        os.makedirs(full_output_dir, exist_ok=True)

    merge_patch_and_save(args.model_suffix, args.patch_path, full_output_dir)


if __name__ == "__main__":
    main()
