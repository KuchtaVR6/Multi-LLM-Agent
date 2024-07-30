import json
import os.path
from collections import defaultdict

from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.patch_utils.patch_sample_collator import PatchAndSampleCollator


class PatchAndSampleCollatorToolbench(PatchAndSampleCollator):
    def __init__(self, patch_manager: PatchManager, planner_prompt_type):
        super(PatchAndSampleCollatorToolbench, self).__init__(patch_manager, planner_prompt_type)
        self.load_file('output_verbose_res/inputs_for_caller.json')

    def load_file(self, file_path):
        with open(file_path, 'rb') as file:
            loaded_samples = json.load(file)

        self.all_samples.extend(loaded_samples)

        if len(loaded_samples) != 0:
            for sample in loaded_samples:
                tool_requested = sample['caller_tool_requested']
                patches = self.patch_manager.return_valid_patches(tool_requested)
                for patch in patches:
                    self.patch_to_samples[patch].append(sample)
