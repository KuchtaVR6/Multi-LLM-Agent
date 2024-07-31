import json

from process_data.toolbench.ood_data_annotation_prompts import categories

# Path to the JSON Lines file
file_path = 'output_verbose_res/outputs_classification.jsonl'
# Path to the output file
output_file_path = 'dataset/toolbench/toolalpaca_api_categories.txt'

def extract_and_save_info(file_path, output_file_path):
    with open(file_path, 'r') as file, open(output_file_path, 'w') as output_file:
        for line in file:
            # Parse the JSON object from each line
            entry = json.loads(line)

            # Extract the API name and category
            api_name = entry.get('custom_id', 'Unknown_API').replace('request-', '', 1)
            category = entry.get('response', {}).get('body', {}).get('choices', [{}])[0].get('message', {}).get(
                'content', 'Unknown_Category')

            # Format the output
            if category not in categories:
                output_line = f'{api_name}: ???\n'
            else:
                output_line = f"{api_name}: {category}\n"

            # Write the output to the file
            output_file.write(output_line)


# Execute the function
extract_and_save_info(file_path, output_file_path)
