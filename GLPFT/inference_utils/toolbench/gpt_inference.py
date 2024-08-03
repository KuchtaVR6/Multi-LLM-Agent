from api_secrets import api_key, base_url
from tqdm import tqdm
import json
import requests
import argparse

model = "gpt-3.5-turbo"

# File paths
input_file_path = 'output_verbose_res/inputs_for_caller.json'
output_file_path = f'output_verbose_res/inferenced_on_{model}.jsonl'

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
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            print(f"Error: Invalid response format. {response.text}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    return None

def process_entries(start_index, input_file_path, output_file_path, api_key, base_url):
    # Initialize aggregate character count and longest message tracking
    total_characters = 0
    max_tokens = 500
    # Open the input file and output file
    with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
        data = json.load(infile)  # Load the JSON data from file

        for entry in tqdm(data[start_index:]):
            prompt = entry.get("model_input_for_caller", "")

            request_body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens
            }

            # Calculate the number of characters in the 'messages' part
            content_length = sum(len(message['content']) for message in request_body['messages'])
            total_characters += content_length

            response = send_real_request(request_body, endpoint=base_url, passkey=api_key)
            entry['predictions'] = '</s>' + response

            # Write the updated entry to the output file
            json.dump(entry, outfile)
            outfile.write('\n')  # Write a newline after each JSON object

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a list of entries from a JSON file.")
    parser.add_argument('--start', type=int, default=0, help='Index of the entry to start processing from')

    args = parser.parse_args()
    start_index = args.start
    print(f"Processing will start from entry index {start_index}.")

    process_entries(start_index, input_file_path, output_file_path, api_key, base_url)

    print('Processing finished')
