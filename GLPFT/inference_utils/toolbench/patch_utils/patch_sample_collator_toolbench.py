from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.patch_utils.patch_sample_collator import PatchAndSampleCollator


class PatchAndSampleCollatorToolbench(PatchAndSampleCollator):
    def __init__(self, patch_manager: PatchManager, planner_prompt_type):
        super(PatchAndSampleCollatorToolbench, self).__init__(patch_manager, planner_prompt_type)
        self.load_file('output_verbose_res/inputs_for_caller.json')
