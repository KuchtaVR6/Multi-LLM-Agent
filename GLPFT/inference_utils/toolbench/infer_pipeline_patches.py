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
from typing import Dict, Optional, Union
import os

from pathlib import Path
from transformers.generation.configuration_utils import GenerationConfig

import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother

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
    planner_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    caller_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    summarizer_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
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
        print("Load data from",data_path)
        temp_data =  json.load(open(data_path, "r"))
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
        input_ids = [self.tokenizer.encode(x) for x in features] # tokens, not pad
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
            input_ids = input_ids,
            attention_mask = attention_mask
        )


def load_plain_model_and_tokenizer(model_name_or_path):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path
    )
    if tokenizer.pad_token_id == None:
        tokenizer.add_special_tokens({"bos_token": "<s>", "eos_token": "</s>", "pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

def infer():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    with open('output_res_partial/toolbench/in_domain/inputs_for_caller.json', 'rb') as file:
        infer_samples_caller = json.load(file)

    # caller inference
    if len(infer_samples_caller) != 0:
        caller_model, caller_tokenizer = load_plain_model_and_tokenizer(model_args.caller_model_name_or_path)
        data_collator = Collator(caller_tokenizer, data_args)
        caller_trainer = TrainerForPred(
            model=caller_model, tokenizer=caller_tokenizer, args=training_args, data_collator=data_collator
        )
        caller_test_dataset = InferDataset([d['model_input_for_caller'] for d in infer_samples_caller], caller_tokenizer, data_args)
        outputs,_ = caller_trainer.predict(caller_test_dataset)
        for i, o in enumerate(outputs):
            candidate = caller_tokenizer.decode(o, skip_special_tokens=True)
            if candidate.startswith(': '):
                candidate = candidate[2:]
            if candidate.strip() in ['','.',',']:
                candidate = 'none'
            infer_samples_caller[i]['predictions'] = "asssitant: " + infer_samples_caller[i]['planner_prediction'] + "</s>caller: " + candidate
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


    final_infer_sample = infer_samples_caller + infer_samples_planner + infer_samples_summarizer
    if process_zero:
        with open(os.path.join(training_args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(final_infer_sample,f, indent=4)



if __name__ == "__main__":
    infer()
