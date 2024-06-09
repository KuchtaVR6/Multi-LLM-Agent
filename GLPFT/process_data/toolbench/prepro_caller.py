import json
import argparse
import os
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
        data =[]
        for f in os.listdir(data_path):
            temp_train = nested_load_data(os.path.join(data_path, f))
            data += temp_train
        return data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        temp_data =  json.load(open(data_path, "r"))
        return temp_data
    else:
        return []


data_paths = args.input_path.split(',')
data= []
for p in data_paths:
    train_temp = nested_load_data(p)
    data += train_temp

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

new_data = []
thought_ambiguous = 0
thought_not_usable = 0

api_counts = defaultdict(int)

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
                thought = d['conversations'][i - 1]['value']  # Get the previous utterance's value

                # Replace placeholders in the query template with history and thought
                input = query_temp.replace('{history}', history).replace('{thought}', thought)

                tool_found = None
                for tool_name in tool_list:
                    if tool_name in thought:
                        if tool_found:
                            thought_ambiguous += 1
                            tool_found = None
                            break
                        else:
                            tool_found = tool_name

                if not tool_found:
                    thought_not_usable += 1
                    continue

                api_counts[tool_found] += 1

                # Add a new entry to 'new_data' with the current tools, history up to the current point, input, and target
                new_data.append({
                    'tools': d['tools'],
                    'history': d['conversations'][:i],
                    'input': input + " caller: ",
                    'target': utter['value']
                })

                # Append the caller's utterance to the history
                history += ('caller: ' + utter['value'] + '</s>')
        elif utter['from'] == 'conclusion':
            history += ('conclusion: ' + utter['value'] + '</s>')

print(len(new_data))
print(thought_ambiguous, thought_not_usable)

# Sort the dictionary by count in descending order
sorted_api_counts = sorted(api_counts.items(), key=lambda item: item[1], reverse=True)

# Print the results line by line
for api, count in sorted_api_counts:
    print(f'{api}: {count}')

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(new_data,f, indent=2)
