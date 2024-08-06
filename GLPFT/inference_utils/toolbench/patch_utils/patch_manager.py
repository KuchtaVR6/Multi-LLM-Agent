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

        print(f'Loading patches from: {root_directory}')

        for dir_path, dir_names, filenames in os.walk(root_directory):
            if 'checkpoint-' in dir_path or dir_path.endswith('_bad_labels'):
                continue  # skip loading checkpoints and models on old data
            if any(file.endswith('.safetensors') for file in filenames):
                # Get the last folder in the chain
                if self.trained_on_all:
                    cut_dir_path = dir_path.rsplit('_', 1)[0]  # remove the 'all' suffix
                last_folder = os.path.basename(cut_dir_path).split('[', 1)[0]
                if self.model_suffix != 'caller':
                    last_folder = last_folder.rsplit('_', 1)[0]  # remove model suffix
                if last_folder.endswith('_merge'):
                    last_folder = last_folder[:-len('_merge')]
                self.dir_path_to_api_name[dir_path] = last_folder

    def categorize_patches(self):
        for dir_path, patch_name in self.dir_path_to_api_name.items():
            hierarchy, patch_type = self.parse_patch_name(patch_name)
            self.patch_hierarchy[hierarchy['category']][hierarchy['api_family']][hierarchy['endpoint']] = dir_path

    def parse_api_categories(self):
        with open('dataset/toolbench/proper_api_categories.txt', 'r') as f:
            for line in f:
                api_fam, category = line.strip().split(': ')
                self.api_to_category[api_fam] = category

        with open('dataset/toolbench/toolalpaca_api_categories.txt', 'r') as f:
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
            'category': category.lower(),
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

    def find_all_merge_adapters(self):
        possible_merges = []

        for category, cat_entries in self.patch_hierarchy.items():
            # check if the category wide is present
            if None in cat_entries and len(cat_entries) >= 2:
                current_candidate_cat = {'lower_order': [], 'higher_order': cat_entries[None][None]}
                for api_family, api_entries in cat_entries.items():
                    if api_family is not None:
                        for endpoint, endpoint_entry in api_entries.items():
                            current_candidate_cat['lower_order'].append(endpoint_entry)
                if len(current_candidate_cat['lower_order']) >= 2:
                    possible_merges.append(current_candidate_cat)

            for api_family, api_entries in cat_entries.items():
                if api_family is None:
                    continue

                if None in api_entries and len(api_entries) >= 3:
                    current_candidate_api = {'lower_order': [], 'higher_order': api_entries[None]}
                    for endpoint, endpoint_entry in api_entries.items():
                        if endpoint is not None:
                            current_candidate_api['lower_order'].append(endpoint_entry)
                    possible_merges.append(current_candidate_api)

        return possible_merges
