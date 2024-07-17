import os
from collections import defaultdict


class PatchManager:
    def __init__(self, model_suffix, trained_on_all=False):
        if not model_suffix:
            self.model_suffix = 'caller'
        else:
            self.model_suffix = model_suffix
        self.trained_on_all = trained_on_all
        self.api_to_category = {}
        self.dir_path_to_api_name = {}
        self.patch_hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        self.parse_api_categories()
        self.categories = [value.lower() for value in self.api_to_category.values()]
        self.find_all_patches()
        self.categorize_patches()

    def find_all_patches(self):
        base_dir = 'output_patches'
        if self.trained_on_all:
            base_dir += '/trained_on_all'
        root_directory = f'{base_dir}/{self.model_suffix}/'

        if not os.path.exists(root_directory):
            raise FileNotFoundError(f"Error: The directory {root_directory} does not exist.")

        for dir_path, dir_names, filenames in os.walk(root_directory):
            if 'checkpoint-' in dir_path:
                continue    # skip loading checkpoints
            if any(file.endswith('.safetensors') for file in filenames):
                # Get the last folder in the chain
                last_folder = os.path.basename(dir_path)
                if self.model_suffix != 'caller':
                    last_folder = last_folder.rsplit('_', 1)[0]  # remove model suffix
                if self.trained_on_all:
                    last_folder = last_folder.rsplit('_', 1)[0]  # remove the 'all' suffix
                self.dir_path_to_api_name[dir_path] = last_folder



    def categorize_patches(self):
        for dir_path, patch_name in self.dir_path_to_api_name.items():
            hierarchy, patch_type = self.parse_patch_name(patch_name)
            self.patch_hierarchy[hierarchy['category']][hierarchy['api_family']][hierarchy['endpoint']] = dir_path

    def parse_api_categories(self):
        with open('dataset/toolbench/api_categories.txt', 'r') as f:
            for line in f:
                api_fam, category = line.strip().split(': ')
                self.api_to_category[api_fam] = category

    def parse_patch_name(self, name):
        if name in self.categories:
            return {
                'category': name,
                'api_family': None,
                'endpoint': None,
            }, 'category'
        if '_for_' in name:
            endpoint, api_family = name.rsplit('_for_', 1)
            patch_type = 'endpoint'
        else:
            endpoint = None
            api_family = name
            patch_type = 'api_family'

        if api_family in self.api_to_category:
            category = self.api_to_category[api_family]
        else:
            category = 'Category not found'

        return {
            'category': category,
            'api_family': api_family,
            'endpoint': endpoint
        }, patch_type

    def return_valid_patches(self, tool_name):
        if tool_name == '[AMBIGUOUS]' or tool_name == None:
            return []
        valid_patches = []
        hierarchy, _ = self.parse_patch_name(tool_name)
        if hierarchy['category'] in self.patch_hierarchy:
            category_entries = self.patch_hierarchy[hierarchy['category']]
            if None in category_entries:
                valid_patches.append(category_entries[None][None])  # category-wide patch
            if hierarchy['api_family'] in category_entries:
                api_family_entries = category_entries[hierarchy['api_family']]
                if None in api_family_entries:
                    valid_patches.append(api_family_entries[None])  # api_family-wide patch
                if hierarchy['endpoint'] in api_family_entries:
                    valid_patches.append(api_family_entries[hierarchy['endpoint']])  # endpoint-specific patch
        return valid_patches

    def all_patch_paths(self):
        return self.dir_path_to_api_name.keys()
