from api_secrets import api_key, base_url

# File paths
input_file_path = 'output_verbose_res/inputs_for_caller.json'
output_file_path = 'output_verbose_res/inference_gpt_requests.jsonl'

# Initialize aggregate character count and longest message tracking
total_characters = 0
max_tokens = 500
num_entries = 0

import json
import requests

def send_real_request(request_body, endpoint, passkey):
    route = "/v1/chat/completions"
    url = f'{endpoint}{route}'
    headers = {
        'Authorization': f'Bearer {passkey}'
    }
    response = requests.post(url, headers=headers, json=request_body)

    if response.status_code == 200:  # Check if the request was successful
        try:
            # Extracting the answer from the JSON response
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"Response: {answer}")
        except (KeyError, IndexError):
            print(f"Error: Invalid response format. {response.text}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    return response

# Open the input file and output file
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    data = json.load(infile)  # Load the JSON data from file
    num_entries = len(data)

    for entry in data:
        identifier = entry.get("caller_sample_id", "unknown")
        prompt = entry.get("model_input_for_caller", "")

        request_body = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens
        }

        # Calculate the number of characters in the 'messages' part
        content_length = sum(len(message['content']) for message in request_body['messages'])
        total_characters += content_length

        print(f"Confirming request for {content_length} characters:")
        confirmation = input("Send request (yes/no): ")
        if confirmation.lower() == 'yes':
            response = send_real_request(request_body, endpoint=base_url, passkey=api_key)
            print(response)

# Print the total number of characters and the longest message content
print(f"Input token estimate: ~{int(total_characters/4)} tokens")
print(f"Output token estimate: ~{int(num_entries*max_tokens/4)} tokens")
