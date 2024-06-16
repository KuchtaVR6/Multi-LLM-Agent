import string
import re
from tqdm import tqdm
import os
import json
from collections import defaultdict


def lower_and_replace_punctuation(text):
    # Convert to lowercase
    text = text.lower()

    # Create a translation table: punctuation -> '_'
    translation_table = str.maketrans(string.punctuation + ' ', '_' * (len(string.punctuation) + 1))

    # Translate the text
    text = text.translate(translation_table)

    # Replace consecutive underscores with a single underscore
    text = re.sub('_+', '_', text)

    return text

def load_api_to_category(file_path):
    api_to_category = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            api_name, category = line.strip().split(': ')
            api_to_category[api_name] = category
    return api_to_category

# Load the API to Category mapping once
api_to_category = load_api_to_category('dataset/toolbench/api_categories.txt')

def find_api_category(api_name):
    category = api_to_category.get(api_name, "Category not found")
    return lower_and_replace_punctuation(category)

# Define directories
input_dir = 'dataset/toolbench/train_per_api/'
output_dir = 'dataset/toolbench/train_per_category/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionaries to store concatenated entries and file counts for each API
category_entries = defaultdict(list)
category_stats = {} # api_count, end_point_count, samples_count per each

skipped = 0

end_point_counts = {}
with open('dataset/toolbench/mergeAPIreport.txt') as file:
    for line in file:
        api_name = line.split(': ')[0]
        second_part = line.split(', ')[1]
        end_point_count = int(second_part.split(' endpoints ')[0])
        end_point_counts[api_name] = end_point_count

# Traverse through all files in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.json'):
        api_name = filename.rsplit('.json', 1)[0]
        category = find_api_category(api_name)

        # Read the JSON file
        with open(os.path.join(input_dir, filename), 'r') as f:
            entries = json.load(f)
            category_entries[category].extend(entries)
            if category not in category_stats:
                category_stats[category] = {
                    'api_count': 0,
                    'end_point_count': 0,
                    'samples_count': 0,
                }
            category_stats[category]['api_count'] += 1
            category_stats[category]['end_point_count'] += end_point_counts[api_name]
            category_stats[category]['samples_count'] += len(entries)

# Write concatenated entries to new JSON files in the output directory
for api_name, entries in tqdm(category_entries.items()):
    output_file = os.path.join(output_dir, f'{api_name}.json')
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=4)

# Print the number of samples and individual endpoints for each API in ascending order
sorted_api_entries = sorted(category_entries.items(), key=lambda item: len(item[1]))
for api_name, entries in sorted_api_entries:
    print(f'{api_name}: {len(entries)} samples, {category_stats[api_name]} endpoints merged')

