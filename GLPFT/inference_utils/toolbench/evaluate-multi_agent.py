import json
import argparse
import re
from rouge import Rouge
import os


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
    if len(ref_list) == 0:
        return 0
    em = 0
    for cand, ref in zip(cand_list, ref_list):
        em += (1 if cand == ref else 0)
    return em/len(cand_list)



def evaluate_action_input_f1(action_pred:list, action_ref: list, cand_list: list, ref_list: list):
    easy_f1 = []
    hard_f1 = []
    f1 = []
    for i in range(len(action_pred)):
        ref_action=action_ref[i]
        pred_action=action_pred[i]

        ref_input = ref_list[i]
        cand_input = cand_list[i]

        if ref_action != pred_action:
            easy_f1.append(0)
            hard_f1.append(0)
            f1.append(0)
        else:
            try:
                ref_input_json = json.loads(ref_input)
                try:
                    cand_input_json = json.loads(cand_input)
                    half_match = 0
                    full_match = 0
                    if ref_input_json == {}:
                        if cand_input_json == {}:
                            easy_f1.append(1)
                            f1.append(1)
                        else:
                            easy_f1.append(0)
                            f1.append(0)
                    else:
                        for k,v in ref_input_json.items():
                            if k in cand_input_json.keys():
                                if cand_input_json[k] == v:
                                    full_match += 1
                                else:
                                    half_match += 1

                        recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                        precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                        hard_f1.append( (2 * recall * precision)/(recall + precision))
                        f1.append( (2 * recall * precision)/(recall + precision))
                except:
                    # cand_input = cand_input.replace("\n","").replace("\"","")
                    # ref_input = cand_input.replace("\n","").replace("\"","")
                    # rouge = Rouge()
                    # rouge_score = rouge.get_scores(hyps=[cand_input], refs=[ref_input], avg=True)
                    if ref_input_json == {}:
                        easy_f1.append(0)
                    else:
                        hard_f1.append(0)
                    # hard_f1.append(rouge_score["rouge-l"]["f"])
                    # f1.append(rouge_score["rouge-l"]["f"])
                    f1.append(0)
            except:
                pass

    return sum(easy_f1) / len(easy_f1)+1e-30, sum(hard_f1) / len(hard_f1)+1e-30, sum(f1) / len(f1)+1e-30


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
        print('='*20)
        print(ref_action, 'vs', pred_action)
        print('-'*20)
        if ref_action == pred_action:
            print('✅ Called Followed expectation')
        else:
            if not pred_action:
                print('🦥 Was expected to called, and didnot')
            elif not ref_action:
                print('🏃 Was expected not to call, and did')
            else:
                print('🤪 Called something else')

            print('-'*20)

            print('ref valid?', evaluate_reasoning(ref_reason, ref_action, tools_available))
            print('pred res valid?', evaluate_reasoning(pred_reason, ref_action, tools_available))
            print('pred reasonable action?', evaluate_reasoning(pred_reason, pred_action, tools_available))

            print('-'*20)
            print(ref_reason)
            print('-'*20)
            print(pred_reason)

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
action_em = evaluate_action_em(action_ref, action_pred)
easy_f1, hard_f1, f1 = evaluate_action_input_f1(action_pred, action_ref, action_input_pred, action_input_ref)
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