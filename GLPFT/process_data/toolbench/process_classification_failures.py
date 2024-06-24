import json
import re
from collections import defaultdict
from operator import itemgetter

with open('classification_fail_log.json') as file:
    entries = json.load(file)

boundary_limit = 3
left_context = defaultdict(int)
right_context = defaultdict(int)

for entry in entries:
    text = re.sub(r'[^\w\s_]', '', entry['edited']).lower()
    text_split = text.split(' ')
    for api in entry['hard_removed']:
        try:
            api_index = text_split.index(api)
            left_bound = max(0, api_index - boundary_limit)
            right_bound = min(len(text_split), api_index + boundary_limit + 1)
            left_context[' '.join(text_split[left_bound:api_index])] += 1
            right_context[' '.join(text_split[api_index:right_bound])] += 1
        except ValueError:
            continue  # Handle case where api is not found in text_split

# Sort dictionaries by value in decreasing order
sorted_left_context = sorted(left_context.items(), key=itemgetter(1), reverse=True)
sorted_right_context = sorted(right_context.items(), key=itemgetter(1), reverse=True)

# Print sorted dictionaries
print("Left Context:")
for context, count in sorted_left_context:
    print(f"{context}: {count}")

print("\nRight Context:")
for context, count in sorted_right_context:
    print(f"{context}: {count}")
