import json
import argparse

import os
from collections import defaultdict
from multi_agent_utils import (evaluate_rougel, evaluate_reasoning, evaluate_action_em, evaluate_action_input_f1,
                               count_unique_strings, display_counts, sort_and_display_dict, parse_output)


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default="")
parser.add_argument('--only_certain', type=bool, default=False)

args = parser.parse_args()


with open(args.input_path, encoding='utf-8') as f:
    data = json.load(f)

plan_ref = []
plan_pred = []
hallu_cases = []
error_cases = []
new_data = []
answer_ref = []
action_ref = []
action_input_ref = []
hallu_ref = 0
answer_pred = []
action_pred = []
action_input_pred = []
hallu_pred = 0

caller_stats = defaultdict(list)
reasoning_stats = defaultdict(list)

for d in data:
    reference = d['reference']
    prediction = d['predictions']

    tools_available = [t['Name'] for t in d['tools']]

    ref_plan, ref_action, ref_input, ref_ans, ref_reason = parse_output(reference)
    pred_plan, pred_action, pred_input, pred_ans, pred_reason = parse_output(prediction)

    if pred_action is not None and pred_action != 'none' and pred_action not in tools_available:
        hallu_pred += 1
        hallu_cases.append(d)
    plan_ref.append(ref_plan)
    plan_pred.append(pred_plan)
    if ref_plan == 'give up':
        pass
    elif ref_plan == 'finish':
        answer_ref.append(ref_ans)
        if pred_ans is None:
            answer_pred.append('none')
        else:
            answer_pred.append(pred_ans)
    else:
        ref_validity = evaluate_reasoning(ref_reason, ref_action, tools_available)
        if ref_validity != 'correct' and args.only_certain:
            continue

        caller_stats['instances'].append(ref_action)
        if ref_action == pred_action:
            caller_stats['correct_action'].append(ref_action)
        else:
            if not pred_action:
                caller_stats['no_action_against_expectation'].append(ref_action)
            elif not ref_action:
                caller_stats['action_against_expectation_of_none'].append(ref_action)
            else:
                caller_stats['wrong_action'].append(ref_action)

            reasoning_stats['ref valid?'].append(ref_validity)
            reasoning_stats['pred res valid?'].append(evaluate_reasoning(pred_reason, ref_action, tools_available))
            reasoning_stats['pred reasonable action?'].append(evaluate_reasoning(pred_reason, pred_action, tools_available))

        action_ref.append(ref_action)
        action_input_ref.append(ref_input)
        if pred_action is None:
            action_pred.append('none')
        else:
            action_pred.append(pred_action)
        
        if pred_input is None:
            action_input_pred.append('{}')
        else:
            action_input_pred.append(pred_input)
        
metric = {}
rouge = evaluate_rougel(answer_pred, answer_ref)
plan_em = evaluate_action_em(plan_ref, plan_pred)

action_em, action_em_per_ref = evaluate_action_em(action_ref, action_pred)
easy_f1_dict, hard_f1_dict, f1_dict = evaluate_action_input_f1(action_pred, action_ref, action_input_pred, action_input_ref)

easy_f1_list = [item for sublist in easy_f1_dict.values() for item in sublist]
hard_f1_list = [item for sublist in hard_f1_dict.values() for item in sublist]
f1_list = [item for sublist in f1_dict.values() for item in sublist]

easy_f1 = sum(easy_f1_list) / len(easy_f1_list)+1e-30
hard_f1 = sum(hard_f1_list) / len(hard_f1_list)+1e-30
f1 = sum(f1_list) / len(f1_list)+1e-30

hallu_rate = hallu_pred / len(data)
metric['rouge'] = rouge
metric['plan_em'] = plan_em
metric['action_em'] = action_em
metric['easy_f1'] = easy_f1
metric['hard_f1'] = hard_f1
metric['f1'] = f1
metric['hallu_rate'] = hallu_rate


if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))        
print(metric)
with open(args.output_path, 'w', encoding='utf-8') as f:
    json.dump(metric,f, indent=2)

with open(args.output_path.replace('metrics.json', 'hallu_cases.json'), 'w', encoding='utf-8') as f:
    json.dump(hallu_cases,f, indent=2)


display_counts(count_unique_strings(caller_stats))
print('---')
display_counts(count_unique_strings(reasoning_stats))
print('---')
sort_and_display_dict(action_em_per_ref, skip_zeros=True)
print('---')

print('F1 INFORMATION')
print('name,em,f1,hard_f1,easy_f1,adj_f1,adj_hard_f1,adj_easy_f1')
for ref in f1_dict.keys():
    f1_data = f1_dict[ref]
    hard_f1_data = hard_f1_dict[ref]
    easy_f1_data = easy_f1_dict[ref]

    f1 = sum(f1_data) / (len(f1_data) + 1e-30)
    hard_f1 = sum(hard_f1_data) / (len(hard_f1_data) + 1e-30)
    easy_f1 = sum(easy_f1_data) / (len(easy_f1_data) + 1e-30)

    if ref in action_em_per_ref:
        em = action_em_per_ref[ref]
    else:
        em = 1

    if em == 0 or em == 1 or f1 + hard_f1 + easy_f1 == 0:
        continue

    print(f'{ref},{em:.2f},{f1:.2f},{hard_f1:.2f},{easy_f1:.2f},'
          f'{f1/em:.2f},{hard_f1/em:.2f},{easy_f1/em:.2f}')
