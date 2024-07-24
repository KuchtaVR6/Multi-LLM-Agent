import json
import random


def select_samples(input_file, output_file, samples_per_target=4):
    with open(input_file, 'r') as file:
        data = json.load(file)

    # Group data by target
    targets = {}
    for item in data:
        target = item['target']
        if target not in targets:
            targets[target] = []
        targets[target].append(item)

    # Select random samples
    selected_samples = []
    for target, items in targets.items():
        if len(items) > samples_per_target:
            selected_samples.extend(random.sample(items, samples_per_target))
        else:
            selected_samples.extend(items)

    # Save to new file
    with open(output_file, 'w') as file:
        json.dump(selected_samples, file, indent=4)


# File paths
input_file = 'dataset/toolbench/new_data/all/category_train.json'
output_file = 'dataset/toolbench/new_data/all/category_toy.json'

# Run the function
select_samples(input_file, output_file)
