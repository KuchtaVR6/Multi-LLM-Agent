import json
import os.path
from collections import defaultdict
from utils.prompt_lib import prompt_dict
import copy


class PatchAndSampleCollator():
    def __init__(self, patch_manager, planner_prompt_type):
        self.patch_manager = patch_manager
        self.patch_to_samples = defaultdict(list)
        self.all_samples = []
        self.planner_prompt_type = planner_prompt_type

    def load_specific_test_sets(self, specific_test_type='certain'):
        specific_test_sets = []
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
            specific_test_sets.append([expert_path, f'{expert_type}/{expert_type}', processed])
        return specific_test_sets

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

    def __iter__(self):
        return PSCIterator(self.patch_to_samples, self.patch_manager.dir_path_to_api_name)


class PSCIterator:
    def __init__(self, patch_to_samples, api_name_lookup):
        self.patch_sample_array = list(patch_to_samples.items())
        self.api_name_lookup = api_name_lookup
        self.count = 0

    def __next__(self):
        if self.count < len(self.patch_sample_array):
            self.count += 1
            return self.prepare_entry(self.count - 1)
        else:
            raise StopIteration

    def prepare_entry(self, index):
        patch_dir, samples = self.patch_sample_array[index]
        api_name = self.api_name_lookup[patch_dir]
        return [patch_dir, api_name, samples]


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

                        tool_selected = None
                        # which tool to use
                        for tool_name in tool_names.split(', '):
                            if tool_name in planner_prediction:
                                if tool_selected:
                                    tool_selected = "[AMBIGUOUS]"
                                    break
                                else:
                                    tool_selected = tool_name

                        reference = u['from'] + ': ' + u['value'] + "</s>" + 'caller' + ": " + d['target'] + '</s>'
                        conversations.append({
                            'tools': d['tools'],
                            'instruction': d['instruction'],
                            'history': c[:j],
                            'dispath': copy.deepcopy(dispatch),
                            'model_input_for_caller': prompt + ' assistant: ',
                            'reference': reference,
                            'caller_sample_id': len(conversations),  # easier identification
                            'caller_tool_requested': tool_selected,
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

# TODO MISSING THE caller_tool_requested
