import json

# File paths
input_file_path = 'output_verbose_res/inputs_for_caller.json'
output_file_path = 'output_verbose_res/inference_gpt_requests.jsonl'

# Initialize aggregate character count and longest message tracking
total_characters = 0
max_tokens = 500
num_entries = 0

# Open the input file and output file
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    data = json.load(infile)  # Load the JSON data from file
    num_entries = len(data)

    for entry in data:
        identifier = entry.get("caller_sample_id", "unknown")
        prompt = entry.get("model_input_for_caller", "")

        # Create the request dictionary with max_tokens
        request = {
            "custom_id": f"request-{identifier}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",  # Updated to gpt-4o-mini
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens  # Added max_tokens
            }
        }

        # Convert request to JSON lines format and write to the output file
        outfile.write(json.dumps(request) + '\n')

        # Calculate the number of characters in the 'messages' part
        for message in request['body']['messages']:
            content_length = len(message['content'])
            total_characters += content_length

# Print the total number of characters and the longest message content
print(f"Input token estimate: ~{int(total_characters/4)} tokens")
print(f"Output token estimate: ~{int(num_entries*max_tokens/4)} tokens")
