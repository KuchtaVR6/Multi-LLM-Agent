import os
import json
from collections import defaultdict

# Define directories
input_dir = 'dataset/toolbench/train_separated/certain/'
output_dir = 'dataset/toolbench/train_per_api/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize dictionaries to store concatenated entries and file counts for each API
api_entries = defaultdict(list)
api_file_count = defaultdict(int)

# Traverse through all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        # Extract the api_name from the filename
        parts = filename.split('_for_')
        if len(parts) != 2:
            continue
        api_name = parts[1].rsplit('.json', 1)[0]

        # Read the JSON file
        with open(os.path.join(input_dir, filename), 'r') as f:
            entries = json.load(f)
            api_entries[api_name].extend(entries)
            api_file_count[api_name] += 1

# Write concatenated entries to new JSON files in the output directory
for api_name, entries in api_entries.items():
    output_file = os.path.join(output_dir, f'{api_name}.json')
    with open(output_file, 'w') as f:
        json.dump(entries, f, indent=4)

# Print the number of samples and individual endpoints for each API in ascending order
sorted_api_entries = sorted(api_entries.items(), key=lambda item: len(item[1]))
for api_name, entries in sorted_api_entries:
    print(f'{api_name}: {len(entries)} samples, {api_file_count[api_name]} endpoints merged')
