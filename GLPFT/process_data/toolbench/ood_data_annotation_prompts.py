import json

# Define the path to the JSON file
file_path = 'ood_output_res_verbose/toolbench/inputs_for_caller.json'

# Load the JSON data from the file
with open(file_path, 'r') as file:
    data = json.load(file)

unique_tools = {}
for entry in data:
    for tool in entry.get('tools', []):
        tool_name = tool.get('Name')
        tool_function = tool.get('function')

        if tool_name and '_for_' in tool_name:
            # Split the tool name
            parts = tool_name.rsplit('_for_', 1)
            if len(parts) == 2:
                name_part, identifier = parts
                identifier = identifier.strip()

                # Extract the description part
                description_start = tool_function.find("The description of this function is:")
                if description_start != -1:
                    description = tool_function[
                                  description_start + len("The description of this function is:"):].strip().strip('"')
                else:
                    description = ""

                # Initialize entry if not present
                if identifier not in unique_tools:
                    unique_tools[identifier] = {
                        "names": set(),
                        "descriptions": set()
                    }

                # Add names and descriptions
                unique_tools[identifier]["names"].add(name_part)
                unique_tools[identifier]["descriptions"].add(description)

categories = ['Logistics', 'Cryptography', 'Jobs', 'Gaming', 'Social', 'Sports', 'Database', 'Business_Software',
              'Music', 'Business', 'Location', 'Travel', 'Artificial_Intelligence_Machine_Learning', 'Science', 'Email',
              'Events', 'Health_and_Fitness', 'Payments', 'Movies', 'Text_Analysis', 'Transportation', 'Monitoring',
              'Medical', 'Financial', 'Weather', 'Video_Images', 'Devices', 'Customized', 'SMS', 'Food',
              'Entertainment', 'Advertising', 'Energy', 'Tools', 'Search', 'Media', 'eCommerce',
              'Visual_Recognition', 'Data', 'Communication', 'Other', 'Finance', 'Cybersecurity', 'News_Media',
              'Translation', 'Mapping', 'Commerce', 'Storage', 'Reward', 'Education', 'Unknown']

char_count = 0
# Create the list of requests
requests = []
for identifier, data in unique_tools.items():
    names = ", ".join(sorted(data["names"]))
    descriptions = " ".join(sorted(data["descriptions"])).strip()
    result = f"{identifier}: {names}|Descriptions: {descriptions}"

    prompt = (f"Classify the API below to one of the categories (your output is only the category name must be in "
              f"the set of categories): {result})\nCategories: {', '.join(categories)}\nCategory: ")

    request = {
        "custom_id": f"request-{identifier}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100
        }
    }

    char_count += len(prompt) + 20
    requests.append(request)

# Save to JSON Lines file
with open('requests.jsonl', 'w') as file:
    for request in requests:
        file.write(json.dumps(request) + '\n')

print(f"Requests saved to requests.jsonl, charcount: {char_count}")