import json
import re
from collections import defaultdict
from operator import itemgetter

with open('classification_fail_log.json') as file:
    entries = json.load(file)

boundary_limit = 3
context = defaultdict(int)

for entry in entries:
    text = re.sub(r'[^\w\s_\[\]\"\'`]', '', entry['edited']).lower()
    text_split = text.split(' ')
    for api in entry['hard_removed']:
        try:
            api_index = text_split.index(api)
            left_bound = max(0, api_index - boundary_limit)
            right_bound = min(len(text_split), api_index + boundary_limit + 1)
            left_context = ' '.join(text_split[left_bound:api_index])
            right_context = ' '.join(text_split[api_index+1:right_bound])
            context[left_context + ' [^ ]+? ' + right_context] += 1
        except ValueError:
            continue  # Handle case where api is not found in text_split

# Sort dictionaries by value in decreasing order
sorted_context = sorted(context.items(), key=itemgetter(1), reverse=True)

# Print sorted dictionaries
print("Context:")
for context, count in sorted_context:
    if count >= 5:
        print(f"{context}: {count}")
