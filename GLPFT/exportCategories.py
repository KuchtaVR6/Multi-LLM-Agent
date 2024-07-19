import os

# Define the root path for the given file structure
root_path = '../data/toolenv/tools'

# Initialize a list to store the formatted lines
formatted_lines = []

# Walk through the directory
for category in os.listdir(root_path):
    category_path = os.path.join(root_path, category)
    if os.path.isdir(category_path):
        for api_family in os.listdir(category_path):
            if '.json' in api_family:
                api_family = api_family.replace('.json', '')
            formatted_lines.append(f"{api_family}: {category}")

# Write the formatted lines to a file
output_file_path = 'dataset/toolbench/proper_api_categories.txt'
with open(output_file_path, 'w') as file:
    for line in formatted_lines:
        file.write(f"{line}\n")
