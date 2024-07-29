import argparse

import numpy as np

from tqdm import tqdm
from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.infer_pipeline_patches import load_model_with_adapters_and_tokenizer

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge adapters for inference.")
    parser.add_argument("model_suffix", type=str, help="The model suffix to be used in the inference script.")
    parser.add_argument("--trained_on_all", action='store_true', help="Use models trained on all samples.")
    args = parser.parse_args()

    model_suffix = args.model_suffix
    trained_on_all = args.trained_on_all

    if not model_suffix:
        model_suffix = 'caller'

    manager = PatchManager(model_suffix, trained_on_all)

    possible_merges = manager.find_all_merge_adapters()
    to_be_merged = []
    all_patches = set()

    for entry in possible_merges:
        print('='*20)
        print('> Can merge: ')
        print('\n'.join(entry['lower_order']))
        print('> Into: ')
        print(entry['higher_order'])
        confirm = input('proceed?')
        if confirm[0].lower() == 'y':
            print('Will be merged')
            name = input('Merged model name:')
            to_be_merged.append({
                'output_name': name,
                'parts': entry['lower_order']
            })

    for merge in tqdm(to_be_merged):
        parts = merge['parts']
        model, token = load_model_with_adapters_and_tokenizer(model_suffix, parts, trained_on_all)
        merge_length = len(parts)
        model.add_weighted_adapter(merge['parts'], np.full(merge_length, 1/merge_length), combination_type="linear",
                                   adapter_name="default")
        for part in parts:
            model.delete_adapter(part)

        output_dir = f'output_patches/{model_suffix}/{merge["output_name"]}'
        if trained_on_all:
            output_dir = output_dir.replace('/', '/trained_on_all/', 1)

        model.save_pretrained(output_dir)


