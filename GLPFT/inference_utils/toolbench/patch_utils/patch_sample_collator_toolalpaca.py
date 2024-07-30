from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.patch_utils.patch_sample_collator import PatchAndSampleCollator


class PatchAndSampleCollatorAlpaca(PatchAndSampleCollator):
    def __init__(self, patch_manager: PatchManager, planner_prompt_type):
        super(PatchAndSampleCollatorAlpaca, self).__init__(patch_manager, planner_prompt_type)
        self.load_file('ood_output_res_verbose/toolbench/inputs_for_caller.json')
