import json
import argparse
import os
import random
from collections import defaultdict
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
for d in data:
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


for api_count_type, dir_path in [[api_counts_certain, all_apis_path], [api_counts_all, certain_apis_path]]:
    for api, cases in api_count_type.items():
        train_cases, test_cases = split_data(cases)

        # Save train cases
        train_file_path = os.path.join(dir_path, f'{api}_train.json')
        with open(train_file_path, 'w', encoding='utf-8') as file:
            json.dump(train_cases, file, indent=2)

        # Save test cases
        test_file_path = os.path.join(dir_path, f'{api}_test.json')
        with open(test_file_path, 'w', encoding='utf-8') as file:
            json.dump(test_cases, file, indent=2)
