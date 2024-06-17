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
from collections import defaultdict
import json
from typing import Dict, Optional, Union
import os

from pathlib import Path
from transformers.generation.configuration_utils import GenerationConfig

import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import PeftConfig

from rouge import Rouge
import gc

from utils.trainer_utils import TrainerForPred


def evaluate_rougel(cand_list: list, ref_list: list):
    if len(ref_list) == 0:
        return 0
    rouge = Rouge()
    rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
    rougel = rouge_score["rouge-l"]["f"]
    return rougel


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_suffix: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    max_input_length: int = field(
        default=1750
    )
    num_infer_samples: int = field(default=-1)
    planner_prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )
    caller_prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )
    summarizer_prompt_type: str = field(
        default='v1', metadata={"help": "the prompt template"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


def nested_load_test_data(data_path):
    test_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_test = nested_load_test_data(os.path.join(data_path, f))
            test_raw_data += temp_test
        return test_raw_data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        print("Load data from", data_path)
        temp_data = json.load(open(data_path, "r"))
        test_raw_data = temp_data
        return test_raw_data
    else:
        return []


class InferDataset(Dataset):
    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, args):
        super(InferDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.args = args

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.raw_data[i]


class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, features):
        input_ids = [self.tokenizer.encode(x) for x in features]  # tokens, not pad
        max_len = max([len(t) for t in input_ids])
        max_len = min(self.args.max_input_length, max_len)
        new_input_ids = []
        for t in input_ids:
            if len(t) > max_len:
                new_t = t[-max_len:]
            else:
                new_t = [self.tokenizer.pad_token_id] * (max_len - len(t)) + t
            new_input_ids.append(new_t)
        input_ids = torch.LongTensor(new_input_ids)
        attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id)
        attention_mask = torch.zeros_like(input_ids).masked_fill(attention_mask, 1)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


def load_plain_model_and_tokenizer(model_suffix, patches_available):

    if not model_suffix:
        model_suffix = 'caller'

    patches_root_directory = f'output_pathes/{model_suffix}/'

    if model_suffix == 'llama':
        model_name_or_path = "meta-llama/Llama-2-7b-hf"
    elif model_suffix == 'backbone':
        model_name_or_path = "saved_models/backbone"
    else:
        model_name_or_path = "saved_models/caller"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    for patch_dir in patches_available.keys():
        current_config = PeftConfig(patches_root_directory + patch_dir)
        model.add_adapter(current_config, adapter_name=patch_dir)

    return model, tokenizer


def find_all_patches(model_suffix):
    if not model_suffix:
        model_suffix = 'caller'

    root_directory = f'output_pathes/{model_suffix}/'

    safetensors_dict = {}

    if not os.path.exists(root_directory):
        raise FileNotFoundError(f"Error: The directory {root_directory} does not exist.")

    for dirpath, dirnames, filenames in os.walk(root_directory):
        if any(file.endswith('.safetensors') for file in filenames):
            # Get the last folder in the chain
            last_folder = os.path.basename(dirpath)
            if model_suffix != 'caller':
                last_folder = last_folder.rsplit('_', 1)[0]  # remove model suffix
            safetensors_dict[dirpath] = last_folder

    return safetensors_dict


def parse_api_categories():
    api_categories = {}
    with open('dataset/toolbench/api_categories.txt', 'r') as f:
        for line in f:
            api_fam, category = line.strip().split(': ')
            api_categories[api_fam] = category
    return api_categories

api_categories = parse_api_categories()

def parse_patch_name(name):
    if name in api_categories.values():
        return {
            'category': name,
            'api_family': None,
            'endpoint': None,
        }, 'category'
    if '_for_' in name:
        endpoint, api_family = name.split('_for_')
        patch_type = 'endpoint'
    else:
        endpoint = None
        api_family = name
        patch_type = 'api_family'
    category = api_categories[api_family]
    return {
        'category': category,
        'api_family': api_family,
        'endpoint': endpoint
    }, patch_type


def categorize_patches(patches_available):
    patch_navigation_map = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
    for dirpath, patch_name in patches_available.items():
        hierarchy, patch_type = parse_patch_name(patch_name)
        patch_navigation_map[hierarchy['category']][hierarchy['api_family']][hierarchy['endpoint']] = dirpath
    return patch_navigation_map


def find_valid_patches(tool_name, categorized_patches):
    valid_patches = []
    hierarchy, _ = parse_patch_name(tool_name)
    if hierarchy['category'] in categorized_patches:
        category_entries = categorized_patches[hierarchy['category']]
        if None in category_entries:
            valid_patches.append(category_entries[None][None])  # category-wide patch
        if hierarchy['api_family'] in category_entries:
            api_family_entries = category_entries[hierarchy['api_family']]
            if None in api_family_entries:
                valid_patches.append(api_family_entries[None])  # api_family-wide patch
            if hierarchy['endpoint'] in api_family_entries:
                valid_patches.append(api_family_entries[hierarchy['endpoint']])  # endpoint-specific patch
    return valid_patches


def transpose_dict(input_dict):
    transposed_dict = {}

    for element, options in input_dict.items():
        for option in options:
            if option not in transposed_dict:
                transposed_dict[option] = []
            transposed_dict[option].append(element)

    return transposed_dict

def infer():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    patches_available = find_all_patches(model_args.model_suffix)

    categorized_patches = categorize_patches(patches_available)

    with open('output_res_partial/toolbench/in_domain/inputs_for_caller.json', 'rb') as file:
        infer_samples_caller = json.load(file)

    # caller inference
    if len(infer_samples_caller) != 0:
        sample_to_patches = {}
        for sample in infer_samples_caller:
            tool_requested = sample['caller_tool_requested']
            patches = find_valid_patches(tool_requested, categorized_patches)
            if patches:
                sample_to_patches[sample] = patches

        patches_to_samples = transpose_dict(sample_to_patches)

        caller_model, caller_tokenizer = load_plain_model_and_tokenizer(model_args.model_suffix, patches_available)
        data_collator = Collator(caller_tokenizer, data_args)
        caller_trainer = TrainerForPred(
            model=caller_model, tokenizer=caller_tokenizer, args=training_args, data_collator=data_collator
        )

        for patch, samples in patches_to_samples.items():
            caller_model.set_adapter(patch)
            caller_test_dataset = InferDataset([d['model_input_for_caller'] for d in samples], caller_tokenizer,
                                               data_args)
            outputs, _ = caller_trainer.predict(caller_test_dataset)
            for i, o in enumerate(outputs):
                candidate = caller_tokenizer.decode(o, skip_special_tokens=True)
                if candidate.startswith(': '):
                    candidate = candidate[2:]
                if candidate.strip() in ['', '.', ',']:
                    candidate = 'none'
                samples[i]['predictions'] = "asssitant: " + samples[i][
                    'planner_prediction'] + "</s>caller: " + candidate

            with open(os.path.join(training_args.output_dir, patch, 'predictions.json'), 'w') as f:
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
    infer()
