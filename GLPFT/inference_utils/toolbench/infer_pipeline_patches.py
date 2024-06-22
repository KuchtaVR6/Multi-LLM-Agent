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
from typing import Optional
import os

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from peft import PeftConfig

import gc

from infer_pipeline import DataArguments, TrainingArguments, InferDataset, Collator
from patch_utils.patch_manager import PatchManager
from utils.trainer_utils import TrainerForPred


IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class ModelArguments:
    model_suffix: Optional[str] = field(default="")


def load_model_with_adapters_and_tokenizer(model_suffix, patch_manager):
    if model_suffix == 'llama':
        model_name_or_path = "meta-llama/Llama-2-7b-hf"
    elif model_suffix == 'backbone':
        model_name_or_path = "saved_models/backbone"
    elif model_suffix == 'dev':
        model_name_or_path = "EleutherAI/pythia-160m"
    else:
        model_name_or_path = "saved_models/caller"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    for patch_dir in patch_manager.all_patch_paths():
        current_config = PeftConfig.from_pretrained(patch_dir)
        model.add_adapter(current_config, adapter_name=patch_dir)

    return model, tokenizer


def transpose_list_of_lists(sample_to_patches):
    transposed_dict = {}

    for sample, patches in sample_to_patches:
        for patch in patches:
            if patch not in transposed_dict:
                transposed_dict[patch] = []
            transposed_dict[patch].append(sample)

    return transposed_dict

def prepare():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    patch_manager = PatchManager(model_args.model_suffix)

    caller_model, caller_tokenizer = load_model_with_adapters_and_tokenizer(model_args.model_suffix, patch_manager)
    data_collator = Collator(caller_tokenizer, data_args)
    caller_trainer = TrainerForPred(
        model=caller_model, tokenizer=caller_tokenizer, args=training_args, data_collator=data_collator
    )

    return patch_manager, caller_model, caller_tokenizer, caller_trainer, data_args, training_args


def infer():
    patch_manager, caller_model, caller_tokenizer, caller_trainer, data_args, training_args = prepare()

    with open('output_verbose_res/inputs_for_caller.json', 'rb') as file:
        infer_samples_caller = json.load(file)

    # caller inference
    if len(infer_samples_caller) != 0:
        sample_to_patches = []
        for sample in infer_samples_caller:
            tool_requested = sample['caller_tool_requested']
            patches = patch_manager.return_valid_patches(tool_requested)
            if patches:
                sample_to_patches.append([sample, patches])

        patches_to_samples = transpose_list_of_lists(sample_to_patches)

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
