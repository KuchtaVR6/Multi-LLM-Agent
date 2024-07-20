import json
from tqdm import tqdm


def load_categories(proper=True):
    categories_path = 'dataset/toolbench/'
    if proper:
        categories_path += 'proper_api_categories.txt'
    else:
        categories_path += 'api_categories.txt'

    # Load the categories file
    with open(categories_path, 'r') as file:
        categories = file.read().splitlines()

    # Create a dictionary for easy lookup
    category_dict = {}
    for category in categories:
        api, category_name = category.split(': ')
        category_dict[api] = category_name

    return category_dict

proper_cat_dict = load_categories()
scraped_cat_dict = load_categories(False)

all_cases = 0
disagreement = 0

for set_type in ['train', 'test']:
    # Paths to the files
    input_json_path = f'dataset/toolbench/new_data/all/{set_type}.json'

    output_json_path = f'dataset/toolbench/new_data/all/category_{set_type}.json'

    # Load the JSON data
    with open(input_json_path, 'r') as file:
        data = json.load(file)

    # Processed data list
    processed_data = []

    for entry in tqdm(data):
        input_data = entry['input']
        target_data = entry['target']

        # Extract the API endpoint from the target string
        api_fam = target_data.split('\n')[0].split('_for_', 1)[1]

        # Lookup the category
        category = proper_cat_dict.get(api_fam, 'Unknown')
        scraped_category = scraped_cat_dict.get(api_fam, 'Unknown')

        if scraped_category != category:
            disagreement += 1

        # Create a new entry
        new_entry = {
            'input': input_data,
            'target': category
        }

        # Add to the processed data list
        processed_data.append(new_entry)

    all_cases += len(data)

    # Save the processed data to a new JSON file
    with open(output_json_path, 'w') as file:
        json.dump(processed_data, file, indent=4)

    print(f'Processed data saved to {output_json_path}')

print(f'Scraped data mistakes {disagreement}/{all_cases}')
