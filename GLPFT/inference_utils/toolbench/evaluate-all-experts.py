from evaluate_expert_improvements import evaluate
import os

import os


def nice_format_output(lines, filename, model_types=2):
    with open(os.path.join(args.output_path, f'{filename}_results.txt'), 'w') as file:
        with open(os.path.join(args.output_path, f'{filename}_results_short.txt'), 'w') as tldr_file:
            def write_both(line):
                file.write(line)
                tldr_file.write(line)

            for adapter_lines in lines:
                write_both('=' * 20 + '\n' + adapter_lines[0] + '\n' + adapter_lines[1] + '\n' + '-' * 20 + '\n')

                rest_of_lines = adapter_lines[2:]

                first_limit = len(rest_of_lines) // model_types

                top_results = ""
                for i in range(model_types):
                    top_results += rest_of_lines[first_limit*i] + '\n'

                write_both(top_results)

                for inner_index in range(1, first_limit):
                    current_results = '-' * 20 + '\n'
                    for i in range(model_types):
                        current_results += rest_of_lines[i*first_limit + inner_index] + '\n'
                    file.write(current_results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_folder', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")
    parser.add_argument('--input_path_backoff', type=str, default="")
    parser.add_argument('--input_path_gpt_backoff', type=str, default="")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # specific datasets
    for test_type in ['all']:
        lines = []

        for dir_path, dir_names, filenames in os.walk(args.input_path_folder):
            current_lines = []

            def output(line):
                current_lines.append(line)


            target_filename = f'{test_type}_expert_predictions.json'
            backoff_filename = f'{test_type}_backoff_predictions.json'
            if 'checkpoint-' in dir_path or dir_path.endswith('_bad_labels') or 'dev' in dir_path:
                continue  # skip loading checkpoints and models on old data
            if any(file.endswith('.safetensors') for file in filenames):
                if target_filename in filenames:
                    if backoff_filename in filenames:
                        base_name = dir_path.rsplit('/', 1)[0].split('/', 1)[1]
                        output(f'Adapter: {dir_path}')
                        evaluate(os.path.join(dir_path, target_filename),
                                 [[base_name, os.path.join(dir_path, backoff_filename)]],
                                 False, False, output_func=output)
                    else:
                        print(f'[MAJOR WARNING] >>> {test_type} BACKOFF MISSING FOR {dir_path}')
                else:
                    print(f'[MAJOR WARNING] >>> {test_type} PREDICTIONS MISSING FOR {dir_path}')

            if len(current_lines) > 2:
                lines.append(current_lines)

        nice_format_output(lines, f'test_specific_{test_type}')

    # toolbench dataset
    lines = []

    for dir_path, dir_names, filenames in os.walk(args.input_path_folder):
        current_lines = []

        def output(line):
            current_lines.append(line)


        target_filename = f'toolbench_expert_predictions.json'
        if target_filename in filenames:
            output(f'Adapter: {dir_path}')
            evaluate(os.path.join(dir_path, target_filename),
                     [
                         ['caller', args.input_path_backoff],
                         # ['gpt4omini', args.input_path_gpt_backoff]
                     ],
                     True, False, output_func=output)

        if len(current_lines) > 2:
            lines.append(current_lines)

    nice_format_output(lines, 'test_toolbench')

    # toolalpaca datasets
    lines = []

    for dir_path, dir_names, filenames in os.walk(args.input_path_folder):
        current_lines = []


        def output(line):
            current_lines.append(line)


        target_filename = f'alpaca_expert_predictions.json'
        backoff_filename = f'alpaca_backoff_predictions.json'
        if 'checkpoint-' in dir_path or dir_path.endswith('_bad_labels') or 'dev' in dir_path:
            continue  # skip loading checkpoints and models on old data
        if any(file.endswith('.safetensors') for file in filenames):
            if target_filename in filenames:
                if backoff_filename in filenames:
                    base_name = dir_path.rsplit('/', 1)[0].split('/', 1)[1]
                    output(f'Adapter: {dir_path}')
                    evaluate(os.path.join(dir_path, target_filename),
                             [[base_name, os.path.join(dir_path, backoff_filename)]],
                             False, False, output_func=output)
                else:
                    print(f'[MAJOR WARNING] >>> alpaca BACKOFF MISSING FOR {dir_path}')
            else:
                print(f'[MAJOR WARNING] >>> alpaca PREDICTIONS MISSING FOR {dir_path}')

        if len(current_lines) > 2:
            lines.append(current_lines)

    nice_format_output(lines, f'test_alpaca_{test_type}')
