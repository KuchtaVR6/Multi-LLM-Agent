import json
from collections import defaultdict


class PatchAndSampleCollator():
    def __init__(self, patch_manager):
        self.patch_manager = patch_manager
        self.patch_to_samples = defaultdict(list)

    def load_file(self, file_path):
        with open(file_path, 'rb') as file:
            infer_samples_caller = json.load(file)

        if len(infer_samples_caller) != 0:
            for sample in infer_samples_caller:
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