import json
import os
from collections import defaultdict

from multi_agent_utils import (evaluate_rougel, evaluate_reasoning, evaluate_action_em, evaluate_action_input_f1,
                               parse_output)


def evaluate(input_path_expert, input_path_backoff, id_sample_matching=False, only_certain=False,
             output_func=print):
    with open(input_path_expert, encoding='utf-8') as f:
        expert_data = json.load(f)

    with open(input_path_backoff, encoding='utf-8') as f:
        if id_sample_matching:
            ids_in = [entry['caller_sample_id'] for entry in expert_data]
            raw_data = json.load(f)
            original_data = [entry for entry in raw_data if entry['caller_sample_id'] in ids_in]
        else:
            original_data = json.load(f)

    output_func('name,em,f1,hard_f1,easy_f1,number_of_samples')
    for [label, current_data] in [['backoff', original_data], ['expert', expert_data]]:
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

        for d in current_data:
            reference = d['reference']
            prediction = d['predictions']
            tools_available = [t['Name'] for t in d['tools']]

            ref_plan, ref_action, ref_input, ref_ans, ref_reason = parse_output(reference)
            pred_plan, pred_action, pred_input, pred_ans, pred_reason = parse_output(prediction)

            if pred_action and pred_action != 'none' and pred_action not in tools_available:
                hallu_pred += 1
                hallu_cases.append(d)
            plan_ref.append(ref_plan)
            plan_pred.append(pred_plan)
            if ref_plan == 'give up':
                pass
            elif ref_plan == 'finish':
                answer_ref.append(ref_ans)
                answer_pred.append(pred_ans if pred_ans else 'none')
            else:
                ref_validity = evaluate_reasoning(ref_reason, ref_action, tools_available)
                if ref_validity != 'correct' and only_certain:
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
                action_pred.append(pred_action if pred_action else 'none')
                action_input_pred.append(pred_input if pred_input else '{}')

        metric = {}
        rouge = evaluate_rougel(answer_pred, answer_ref)
        plan_em, _ = evaluate_action_em(plan_ref, plan_pred)

        action_em, action_em_per_ref = evaluate_action_em(action_pred, action_ref)
        easy_f1_dict, hard_f1_dict, f1_dict = evaluate_action_input_f1(action_pred, action_ref, action_input_pred, action_input_ref)

        easy_f1_list = [item for sublist in easy_f1_dict.values() for item in sublist]
        hard_f1_list = [item for sublist in hard_f1_dict.values() for item in sublist]
        f1_list = [item for sublist in f1_dict.values() for item in sublist]

        easy_f1 = sum(easy_f1_list) / (len(easy_f1_list) + 1e-30)
        hard_f1 = sum(hard_f1_list) / (len(hard_f1_list) + 1e-30)
        f1 = sum(f1_list) / (len(f1_list) + 1e-30)

        hallu_rate = hallu_pred / len(current_data)
        metric['rouge'] = rouge
        metric['plan_em'] = plan_em
        metric['action_em'] = action_em
        metric['easy_f1'] = easy_f1
        metric['hard_f1'] = hard_f1
        metric['f1'] = f1
        metric['hallu_rate'] = hallu_rate

        output_func(f'{label},{action_em:.2f},{f1:.2f},{hard_f1:.2f},{easy_f1:.2f},{len(f1_list)}')

        for ref in f1_dict.keys():
            f1_data = f1_dict[ref]
            hard_f1_data = hard_f1_dict[ref]
            easy_f1_data = easy_f1_dict[ref]

            f1 = sum(f1_data) / (len(f1_data) + 1e-30)
            hard_f1 = sum(hard_f1_data) / (len(hard_f1_data) + 1e-30)
            easy_f1 = sum(easy_f1_data) / (len(easy_f1_data) + 1e-30)

            em = action_em_per_ref.get(ref, 1)

            output_func(f'{label}_{ref},{em:.2f},{f1:.2f},{hard_f1:.2f},{easy_f1:.2f},{len(f1_data)}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_expert', type=str, default="")
    parser.add_argument('--input_path_backoff', type=str, default="")
    parser.add_argument('--id_sample_matching', type=bool, default=False)
    parser.add_argument('--only_certain', type=bool, default=False)

    args = parser.parse_args()

    evaluate(args.input_path_expert, args.input_path_backoff, args.id_sample_matching, args.only_certain)
