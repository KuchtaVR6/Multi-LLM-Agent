import argparse
from inference_utils.toolbench.patch_utils.patch_manager import PatchManager

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Merge adapters for inference.")
    parser.add_argument("model_suffix", type=str, help="The model suffix to be used in the inference script.")
    parser.add_argument("--trained_on_all", action='store_true', help="Use models trained on all samples.")
    args = parser.parse_args()

    model_suffix = args.model_suffix
    trained_on_all = args.trained_on_all

    manager = PatchManager(model_suffix, trained_on_all)

    manager.find_all_merge_adapters()

