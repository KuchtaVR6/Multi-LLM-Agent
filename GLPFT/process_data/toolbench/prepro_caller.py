import json
import argparse
import os
import random
from collections import defaultdict

from tqdm import tqdm
import string
import re
from utils.prompt_lib import prompt_dict

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--prompt_type', type=str)

args = parser.parse_args()

prompt_temp = prompt_dict[args.prompt_type]

print("####################### PREPRO CALLER DATA #####################")


def str2bool(text):
    if text.lower() in ['true', 't', '1']:
        return True
    else:
        return False


def nested_load_data(data_path):
    if os.path.isdir(data_path):
        data = []
        for f in os.listdir(data_path):
            temp_train = nested_load_data(os.path.join(data_path, f))
            data += temp_train
        return data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        temp_data = json.load(open(data_path, "r"))
        return temp_data
    else:
        return []


data_paths = args.input_path.split(',')
data = []
for p in data_paths:
    train_temp = nested_load_data(p)
    data += train_temp

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

tool_utterance = 0
thought_ambiguous = 0
thought_not_usable = 0

planner_caller_mismatch = 0

api_counts_certain = defaultdict(list)
api_counts_all = defaultdict(list)

# Loop through each item in the 'data' list
for d in tqdm(data):
    tool_docs = ""  # Initialize an empty string to hold the tool documentation

    # Loop through each tool in the current data item
    for t in d['tools']:
        tool_docs += json.dumps(t) + '\n'  # Append the JSON representation of the tool to 'tool_docs'

    # Create a comma-separated string of tool names
    tool_list = [t['Name'] for t in d['tools']]
    tool_names = ', '.join(tool_list)

    # Replace placeholders in the prompt template with the actual tool documentation and tool names
    query_temp = prompt_temp.replace('{doc}', tool_docs).replace('{tool_names}', tool_names)

    history = ""  # Initialize an empty string to hold the conversation history

    # Loop through each conversation in the current data item
    for i in range(len(d['conversations'])):
        utter = d['conversations'][i]  # Get the current utterance

        # Append the utterance to the history based on its 'from' field
        if utter['from'] == 'assistant':
            history += ('assistant: ' + utter['value'] + '</s>')
        elif utter['from'] == 'user':
            history += ('user: ' + str(utter['value']) + '</s>')
        elif utter['from'] == 'observation':
            history += ('observation: ' + utter['value'])
        elif utter['from'] == 'caller':
            # Check for 'invalid_hallucination_function_name' in the caller's value
            if 'invalid_hallucination_function_name' in utter['value']:
                pass  # Skip if the condition is met
            else:
                tool_utterance += 1
                thought = d['conversations'][i - 1]['value']  # Get the previous utterance's value

                # Replace placeholders in the query template with history and thought
                input = query_temp.replace('{history}', history).replace('{thought}', thought)

                mentioned_tool = None
                for tool_name in tool_list:
                    if tool_name in thought:
                        if mentioned_tool:
                            thought_ambiguous += 1
                            mentioned_tool = None
                            break
                        else:
                            mentioned_tool = tool_name

                tool_used_after = utter['value'].split('\n')[0].split(' ')[1]

                utterance = {
                    'tools': d['tools'],
                    'history': d['conversations'][:i],
                    'input': input + " caller: ",
                    'target': utter['value']
                }

                api_counts_all[tool_used_after].append(utterance)

                if not mentioned_tool:
                    thought_not_usable += 1
                    continue

                if tool_used_after != mentioned_tool:
                    planner_caller_mismatch += 1
                    continue

                api_counts_certain[mentioned_tool].append(utterance)

                # Append the caller's utterance to the history
                history += ('caller: ' + utter['value'] + '</s>')
        elif utter['from'] == 'conclusion':
            history += ('conclusion: ' + utter['value'] + '</s>')

all_apis_path = os.path.dirname(args.output_path + 'all/')

if not os.path.exists(all_apis_path):
    os.makedirs(all_apis_path)

certain_apis_path = os.path.dirname(args.output_path + 'certain/')

if not os.path.exists(certain_apis_path):
    os.makedirs(certain_apis_path)


# Function to split the data
def split_data(data, test_size=0.1):
    """Splits data into train and test sets."""
    random.shuffle(data)
    split_index = int(len(data) * (1 - test_size))
    return data[:split_index], data[split_index:]


api_name_seperator = '_for_'

def load_api_to_category(file_path):
    api_to_category = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            api_name, category = line.strip().split(': ')
            api_to_category[api_name] = category
    return api_to_category

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

# Load the API to Category mapping once
api_to_category = load_api_to_category('dataset/toolbench/api_categories.txt')

def find_api_category(api_name):
    category = api_to_category.get(api_name, "Category not found")
    return lower_and_replace_punctuation(category)

for api_count_type, dir_path in [[api_counts_certain, all_apis_path], [api_counts_all, certain_apis_path]]:
    api_entries_train = defaultdict(list)
    api_entries_test = defaultdict(list)
    category_entries_train = defaultdict(list)
    category_entries_test = defaultdict(list)

    for final_folder in ['endpoint', 'api_family', 'category']:
        if not os.path.exists(dir_path + '/' + final_folder):
            os.makedirs(dir_path + '/' + final_folder)

    for api, cases in tqdm(api_count_type.items()):
        if api_name_seperator not in api:
            continue  # exclude the ones that are not conforming to the syntax
        train_cases, test_cases = split_data(cases)

        # Save train cases
        train_file_path = os.path.join(dir_path, 'endpoint/', f'{api}_train.json')
        with open(train_file_path, 'w', encoding='utf-8') as file:
            json.dump(train_cases, file, indent=2)

        # Save test cases
        test_file_path = os.path.join(dir_path, 'endpoint/', f'{api}_test.json')
        with open(test_file_path, 'w', encoding='utf-8') as file:
            json.dump(test_cases, file, indent=2)

        endpoint, api_name = api.rsplit(api_name_seperator, 1)
        api_entries_train[api_name].extend(train_cases)
        api_entries_test[api_name].extend(test_cases)

        category = find_api_category(api_name)

        category_entries_train[category].extend(train_cases)
        category_entries_test[category].extend(test_cases)

    # Write concatenated entries to new JSON files in the output directory
    for api_name, entries in tqdm(api_entries_train.items()):
        output_file = os.path.join(dir_path, 'api_family/', f'{api_name}_train.json')
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)

    # Write concatenated entries to new JSON files in the output directory
    for api_name, entries in tqdm(api_entries_test.items()):
        output_file = os.path.join(dir_path, 'api_family/', f'{api_name}_test.json')
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)

    # Write concatenated entries to new JSON files in the output directory
    for category_name, entries in tqdm(category_entries_train.items()):
        output_file = os.path.join(dir_path, 'category/', f'{category_name}_train.json')
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)

    # Write concatenated entries to new JSON files in the output directory
    for category_name, entries in tqdm(category_entries_test.items()):
        output_file = os.path.join(dir_path, 'category/', f'{category_name}_test.json')
        with open(output_file, 'w') as f:
            json.dump(entries, f, indent=4)
