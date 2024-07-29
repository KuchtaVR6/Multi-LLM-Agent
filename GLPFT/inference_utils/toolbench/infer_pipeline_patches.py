# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.


from dataclasses import dataclass, field
import json
from typing import Optional
import os
from tqdm import tqdm

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import PeftConfig, get_peft_model
from supportedModels import get_model_path_on_suffix

import gc

from inference_utils.toolbench.infer_pipeline import DataArguments, TrainingArguments, InferDataset, Collator
from inference_utils.toolbench.patch_utils.patch_manager import PatchManager
from inference_utils.toolbench.patch_utils.patch_sample_collator import PatchAndSampleCollator
from utils.trainer_utils import TrainerForPred

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class TestArguments:
    model_suffix: Optional[str] = field(default="")
    trained_on_all: Optional[bool] = field(default=False)
    regular_test_set: Optional[bool] = field(default=True)
    test_backoff: Optional[bool] = field(default=False)
    do_specific_tests: Optional[bool] = field(default=False)
    do_specific_tests_backoff: Optional[bool] = field(default=True)
    specific_test_sets: Optional[str] = field(default="certain")


def load_model_with_adapters_and_tokenizer(model_suffix, list_of_patch_paths, trained_on_all=False):
    model_name_or_path = get_model_path_on_suffix(model_suffix, trained_on_all)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    peftified_models = False

    for patch_dir in tqdm(list_of_patch_paths):
        current_config = PeftConfig.from_pretrained(patch_dir)
        if not peftified_models:
            model = get_peft_model(model, current_config, adapter_name=patch_dir)
            peftified_models = True
        else:
            model.add_adapter(patch_dir, current_config)

    return model, tokenizer


def infer_on_samples(samples, trainer, tokenizer, data_args):
    caller_test_dataset = InferDataset([d['model_input_for_caller'] for d in samples], tokenizer,
                                       data_args)
    outputs, _ = trainer.predict(caller_test_dataset)
    for i, o in enumerate(outputs):
        candidate = tokenizer.decode(o, skip_special_tokens=True)
        if candidate.startswith(': '):
            candidate = candidate[2:]
        if candidate.strip() in ['', '.', ',']:
            candidate = 'none'
        samples[i]['predictions'] = "assistant: " + samples[i][
            'planner_prediction'] + "</s>caller: " + candidate

    return samples


def infer(input_files):
    parser = transformers.HfArgumentParser(
        (TestArguments, DataArguments, TrainingArguments)
    )
    test_args, data_args, training_args = parser.parse_args_into_dataclasses()

    patch_manager = PatchManager(test_args.model_suffix, test_args.trained_on_all)

    caller_model, caller_tokenizer = load_model_with_adapters_and_tokenizer(test_args.model_suffix,
                                                                            patch_manager.all_patch_paths(),
                                                                            test_args.trained_on_all)

    data_collator = Collator(caller_tokenizer, data_args)
    caller_trainer = TrainerForPred(
        model=caller_model, tokenizer=caller_tokenizer, args=training_args, data_collator=data_collator
    )

    collator = PatchAndSampleCollator(patch_manager, data_args.planner_prompt_type)
    for input_file in input_files:
        collator.load_file(input_file)

    if test_args.regular_test_set:
        print('Predicting the ToolBench Test set...')
        for patch, api_name, samples in collator:
            target_filepath = os.path.join(patch, f'toolbench_expert_predictions.json')
            if os.path.exists(target_filepath) and os.path.getsize(target_filepath) > 0:
                continue
            caller_model.set_adapter(patch)
            samples = infer_on_samples(samples, caller_trainer, caller_tokenizer, data_args)

            with open(target_filepath, 'w') as f:
                json.dump(samples, f, indent=4)

    if test_args.do_specific_tests:
        print('Predicting the Expert Specific Test sets...')
        for patch, api_name, samples in collator.load_specific_test_sets(test_args.specific_test_sets):
            target_filepath = os.path.join(patch, f'{test_args.specific_test_sets}_expert_predictions.json')
            if os.path.exists(target_filepath) and os.path.getsize(target_filepath) > 0:
                continue
            caller_model.set_adapter(patch)
            samples = infer_on_samples(samples, caller_trainer, caller_tokenizer, data_args)

            with open(target_filepath, 'w') as f:
                json.dump(samples, f, indent=4)

    # remove all adapters from the model before the backoff tests
    caller_model.disable_adapter_layers()
    for adapter in patch_manager.all_patch_paths():
        caller_model.delete_adapter(adapter)

    if test_args.test_backoff:
        print('Predicting the Toolbench Test sets on backoff...')
        samples = infer_on_samples(collator.all_samples, caller_trainer, caller_tokenizer, data_args)
        os.makedirs(training_args.output_dir, exist_ok=True)

        with open(os.path.join(training_args.output_dir, 'toolbench_backoff_predictions.json'), 'w') as f:
            json.dump(samples, f, indent=4)

    if test_args.do_specific_tests_backoff:
        print('Predicting the Expert Specific Test sets on backoff...')
        for patch, api_name, samples in collator.load_specific_test_sets(test_args.specific_test_sets):
            target_filepath = os.path.join(patch, f'{test_args.specific_test_sets}_backoff_predictions.json')
            if os.path.exists(target_filepath) and os.path.getsize(target_filepath) > 0:
                continue

            samples = infer_on_samples(samples, caller_trainer, caller_tokenizer, data_args)
            with open(target_filepath, 'w') as f:
                json.dump(samples, f, indent=4)

    caller_model.to('cpu')
    caller_trainer.model.to('cpu')
    caller_trainer.model_wrapped.to('cpu')
    del caller_model
    del caller_tokenizer
    del caller_trainer.model
    del caller_trainer.model_wrapped
    del caller_trainer
    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    infer(['output_verbose_res/inputs_for_caller.json'])
