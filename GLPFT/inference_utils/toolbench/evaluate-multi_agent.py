import json
import argparse
import re

import numpy as np
from rouge import Rouge
import os
from collections import defaultdict, Counter


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default="")
parser.add_argument('--output_path', type=str, default="")

args = parser.parse_args()


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score["rouge-l"]["f"]
    return rougel

def evaluate_action_em(cand_list: list, ref_list: list):
    print(cand_list, ref_list)
    if len(ref_list) == 0:
        return 0
    em = 0
    em_per_ref = defaultdict(list)
    for cand, ref in zip(cand_list, ref_list):
        em_val = (1 if cand == ref else 0)
        em += em_val
        em_per_ref[ref].append(em_val)

    em_final = {}
    for ref, ems in em_per_ref.items():
        em_final[ref] = np.average(ems)

    return em/len(cand_list), em_final



def evaluate_action_input_f1(action_pred:list, action_ref: list, cand_list: list, ref_list: list):
    easy_f1 = defaultdict(list)
    hard_f1 = defaultdict(list)
    f1 = defaultdict(list)
    for i in range(len(action_pred)):
        ref_action=action_ref[i]
        pred_action=action_pred[i]

        ref_input = ref_list[i]
        cand_input = cand_list[i]

        if ref_action != pred_action:
            easy_f1[ref_action].append(0)
            hard_f1[ref_action].append(0)
            f1[ref_action].append(0)
        else:
            try:
                ref_input_json = json.loads(ref_input)
                try:
                    cand_input_json = json.loads(cand_input)
                    half_match = 0
                    full_match = 0
                    if ref_input_json == {}:
                        if cand_input_json == {}:
                            easy_f1[ref_action].append(1)
                            f1[ref_action].append(1)
                        else:
                            easy_f1[ref_action].append(0)
                            f1[ref_action].append(0)
                    else:
                        for k,v in ref_input_json.items():
                            if k in cand_input_json.keys():
                                if cand_input_json[k] == v:
                                    full_match += 1
                                else:
                                    half_match += 1

                        recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                        precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                        hard_f1[ref_action].append( (2 * recall * precision)/(recall + precision))
                        f1[ref_action].append( (2 * recall * precision)/(recall + precision))
                except:
                    # cand_input = cand_input.replace("\n","").replace("\"","")
                    # ref_input = cand_input.replace("\n","").replace("\"","")
                    # rouge = Rouge()
                    # rouge_score = rouge.get_scores(hyps=[cand_input], refs=[ref_input], avg=True)
                    if ref_input_json == {}:
                        easy_f1[ref_action].append(0)
                    else:
                        hard_f1[ref_action].append(0)
                    # hard_f1.append(rouge_score["rouge-l"]["f"])
                    # f1.append(rouge_score["rouge-l"]["f"])
                    f1[ref_action].append(0)
            except:
                pass

    return easy_f1, hard_f1, f1


with open(args.input_path, encoding='utf-8') as f:
    data = json.load(f)



def parse_output(text):
    end_of_reasoning = len(text)
    if 'Next: ' in text:
        end_of_reasoning = text.rindex('Next: ')
    start_of_reasoning = 0
    starters = ['asssitant: ', 'assistant: ']
    for starter in starters:
        if text.startswith(starter):
            start_of_reasoning = len(starter)

    prev_reasoning = text[start_of_reasoning:end_of_reasoning]

    if 'Next: give up' in text:
        return "give up", None, None, None, prev_reasoning
    elif 'Next: conclusion' in text:
        # finish
        plan = 'finish'
        action = None
        action_input = None
        if 'conclusion:' in text:
            answer_idx = text.rindex('conclusion: ')
            answer = text[answer_idx + len('conclusion: '):]
        else:
            answer = text
        answer = answer.replace('</s>',"")
        return plan, action, action_input, answer, prev_reasoning
    else :
        plan='call'
        text = text.split('</s>')[1]
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        answer = None
        return plan, action, action_input, answer, prev_reasoning

def evaluate_reasoning(reasoning, expected_api, apis):
    number_of_apis_mentioned = 0
    for api in apis:
        if api in reasoning:
            number_of_apis_mentioned += 1

    if expected_api:
        correct_api_mentioned = expected_api in reasoning
        if correct_api_mentioned:
            if number_of_apis_mentioned == 1:
                return 'correct'
            else:
                return 'ambiguous'
        else:
            if number_of_apis_mentioned == 0:
                return 'no_apis'
            else:
                return 'wrong_apis'
    else:
        if number_of_apis_mentioned == 0:
            return 'partially_correct_no_mention'
        else:
            return 'apis_present_but_not_expected'

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

            reasoning_stats['ref valid?'].append(evaluate_reasoning(ref_reason, ref_action, tools_available))
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


def count_unique_strings(data_dict):
    result = defaultdict(Counter)
    for key, string_list in data_dict.items():
        result[key] = Counter(string_list)
    return result

def display_counts(count_dict):
    for key, counts in count_dict.items():
        total_count = sum(counts.values())
        print(f"Category: {key} (Total: {total_count})")
        sort_and_display_dict(counts)
        print()  # Blank line for better readability

def sort_and_display_dict(input_dict, skip_zeros=False):
    sorted_counts = sorted(input_dict.items(), key=lambda item: item[1], reverse=True)
    for item, count in sorted_counts:
        if count == 0 and skip_zeros:
            break  # because its sorted (continue would work too)
        print(f"  {item}: {count}")
    print('WARN - All zero values skipped')

display_counts(count_unique_strings(caller_stats))
print('---')
display_counts(count_unique_strings(reasoning_stats))
print('---')
sort_and_display_dict(action_em_per_ref, skip_zeros=True)
print('---')