import copy
import json
import os.path

from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.patch_utils.patch_sample_collator import PatchAndSampleCollator, PSCIterator
from utils.prompt_lib import prompt_dict
from utils.tool_classifier import ToolClassifier


class PatchAndSampleCollatorSpecific(PatchAndSampleCollator):
    def __init__(self, patch_manager: PatchManager, planner_prompt_type, specific_test_type='certain'):
        super(PatchAndSampleCollatorSpecific, self).__init__(patch_manager, planner_prompt_type)
        folder_path = f'dataset/toolbench/new_data/'
        for expert_path in self.patch_manager.all_patch_paths():
            expert_name = self.patch_manager.dir_path_to_api_name[expert_path]
            _, expert_type = self.patch_manager.parse_patch_name(expert_name)
            specified_folder_path = f'{specific_test_type}/{expert_type}/'
            filename = f'{expert_name}_test.json'
            full_path = os.path.join(folder_path, specified_folder_path, filename)
            with open(full_path, 'rb') as file:
                loaded_samples = json.load(file)
            processed = build_caller_infer_samples(loaded_samples, self.planner_prompt_type)
            self.patch_to_samples[expert_path] = processed

def build_caller_infer_samples(raw_data, planner_prompt_type):
    conversations = []
    prompt_temp = prompt_dict[planner_prompt_type]
    for d in raw_data:
        c = d['history']
        d['instruction'] = None
        tool_docs = ""
        for t in d['tools']:
            tool_docs += json.dumps(t) + '\n'
        tool_names = ', '.join([t['Name'] for t in d['tools']])
        query_temp = prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}', tool_names)
        dispatch = ""
        for j, u in enumerate(c):
            if u['from'] == 'assistant':
                if "Next: caller." in u['value'] or "Next: conclusion." in u['value'] or "Next: give up." in u['value']:
                    prompt = query_temp.replace('{history}', dispatch)
                    if j + 1 == len(c) and "Next: caller" in u['value']:
                        action_end_idx = u['value'].index("Next: caller.")
                        planner_prediction = u['value'][:action_end_idx + len("Next: caller.")]

                        tool_classifier = ToolClassifier(tool_names.split(', '))
                        mentioned_tool = tool_classifier.feed_plan(planner_prediction)

                        print('=='*20)
                        print(tool_names.split(', '),'->',mentioned_tool)
                        print('--'*20)
                        print(planner_prediction)
                        print('=='*20)

                        reference = u['from'] + ': ' + u['value'] + "</s>" + 'caller' + ": " + d['target'] + '</s>'
                        conversations.append({
                            'tools': d['tools'],
                            'instruction': d['instruction'],
                            'history': c[:j],
                            'dispath': copy.deepcopy(dispatch),
                            'model_input_for_caller': d['input'],
                            'reference': reference,
                            'caller_sample_id': len(conversations),  # easier identification
                            'caller_tool_requested': mentioned_tool,
                            'planner_prediction': planner_prediction
                        })
                dispatch += ('assistant: ' + u['value'] + '</s>')
            elif u['from'] == 'user':
                if not d['instruction']:
                    d['instruction'] = u['value']
                dispatch += ('user: ' + u['value'] + '</s>')
            elif u['from'] == 'observation':
                dispatch += ('observation: ' + u['value'])
            elif u['from'] == 'caller':
                dispatch += ('caller: ' + u['value'] + '</s>')
            elif u['from'] == 'conclusion':
                dispatch += ('conclusion: ' + u['value'] + '</s>')

    return conversations
