from evaluate_expert_improvements import evaluate
import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path_folder', type=str, default="")
    parser.add_argument('--output_path', type=str, default="")

    args = parser.parse_args()

    # specific datasets

    os.makedirs(args.output_path, exist_ok=True)

    for test_type in ['all', 'certain']:
        with open(os.path.join(args.output_path, f'test_specific_results_{test_type}.txt'), 'w') as file:
            def output(line):
                file.write(line)

            for dir_path, dir_names, filenames in os.walk(args.input_path_folder):
                target_filename = f'{test_type}_expert_predictions.json'
                backoff_filename = f'{test_type}_backoff_predictions.json'
                if target_filename in filenames:
                    if backoff_filename in filenames:
                        file.write(f'Adapter: {dir_path}')
                        evaluate(os.path.join(dir_path, target_filename),
                                 os.path.join(dir_path, backoff_filename), args.output_path,
                                 False, False, output_func=output)
                    else:
                        print(f'[MAJOR WARNING] >>> BACKOFF MISSING FOR {dir_path}')

    # toolbench dataset

    with open(os.path.join(args.output_path, f'test_toolbench_results.txt'), 'w') as file:
        def output(line):
            file.write(line)

        for dir_path, dir_names, filenames in os.walk(args.input_path_folder):
            target_filename = f'toolbench_expert_predictions.json'
            if target_filename in filenames:
                file.write(f'Adapter: {dir_path}')
                evaluate(os.path.join(dir_path, target_filename), args.input_path_backoff, args.output_path,
                         False, False, output_func=output)


