import json

# Load the original data (inputs)
with open('output_verbose_res/inputs_for_caller.json', 'r') as f:
    original_data = json.load(f)

# Load the predictions data
predictions = []
with open('output_verbose_res/gpt_outputs.jsonl', 'r') as f:
    for line in f:
        predictions.append(json.loads(line))

# Create a mapping of custom_id to prediction content
prediction_map = {}
for pred in predictions:
    custom_id = pred.get('custom_id')
    content = pred.get('response', {}).get('body', {}).get('choices', [{}])[-1].get('message', {}).get('content')
    if custom_id and content:
        prediction_map[custom_id.replace('request-','')] = content

# Add predictions to the original data
for entry in original_data:
    caller_sample_id = str(entry.get('caller_sample_id'))
    if caller_sample_id in prediction_map:
        entry['predictions'] = '</s>' + prediction_map[caller_sample_id]
    else:
        print("Prediction missing - 😲")
        entry['predictions'] = None  # Or handle cases where there is no prediction

output_file = 'output_verbose_res/gpt_full_outputs.json'

# Write the updated data to a new JSON file
with open(output_file, 'w') as f:
    json.dump(original_data, f, indent=4)

print(f"Processing complete. Updated data has been saved to '{output_file}'.")
