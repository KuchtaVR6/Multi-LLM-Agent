import json
import os
import random
from typing import Dict
from torch.utils.data import Dataset

import torch
import transformers
from transformers.trainer_pt_utils import LabelSmoother


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    data_paths = data_args.data_path.split(',')
    train_raw_data = []
    eval_raw_data = []
    for p in data_paths:
        train_temp, dev_temp = nested_load_data(p, data_args.max_num_sample_per_data, data_args.max_num_sample_ratio)
        train_raw_data += train_temp
        eval_raw_data += dev_temp

    # prompt_temp = prompt_dict["v7_" + data_args.prompt_type]
    # gorilla_prompt_temp = prompt_dict["v7_gorilla_" + data_args.prompt_type]


    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template("collab_agent_v3")
    # roles = {"user": conv.roles[0], "assistant": conv.roles[1], "caller": conv.roles[2], 'observation':conv.roles[3], 'conclusion': conv.roles[4]}

    # Apply prompt templates
    datas = []
    for i, source in enumerate(sources):
        input = 'system: ' + source['input']
        input_ids = tokenizer.encode(input) #list
        target = source['target'] + "</s>"
        target_ids = tokenizer.encode(target)[1:]
        instruction_len = len(input_ids)
        answer_len = len(target_ids)
        input_ids = input_ids + target_ids

        if len(input_ids) < tokenizer.model_max_length:
            input_ids += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_ids))
            input_ids = torch.LongTensor(input_ids).cpu()
            labels = input_ids.clone().cpu()
            labels[:instruction_len] = IGNORE_TOKEN_ID
            labels[instruction_len + answer_len:] = IGNORE_TOKEN_ID
            # print(input_ids.detach().cpu().numpy().tolist())
            # print('---------------------')
            # print(targets.detach().cpu().numpy().tolist())

        else:
            input_ids = torch.LongTensor(input_ids).cpu()
            labels = input_ids.clone().cpu()
            labels[:instruction_len] = IGNORE_TOKEN_ID
            input_ids = input_ids[-tokenizer.model_max_length:]
            labels = labels[-tokenizer.model_max_length:]
        input_ids.requires_grad=False
        labels.requires_grad=False

        datas.append(dict(
            input_ids=input_ids.cpu(),
            labels=labels.cpu(),
            attention_mask=input_ids.ne(tokenizer.pad_token_id).cpu(),
        ))

    return datas


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data,tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        sources = [example for example in raw_data]
        data_dict = preprocess(sources, tokenizer)
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.data_dict[i]


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if 'conversations' in self.raw_data[i].keys():
            ret = preprocess([self.raw_data[i]], self.tokenizer)
        else:
            ret = preprocess([self.raw_data[i]], self.tokenizer)

        ret = dict(
            input_ids=ret[0]["input_ids"],
            labels=ret[0]["labels"],
            attention_mask=ret[0]["attention_mask"],
        )
        self.cached_data_dict[i] = ret

        return ret

def nested_load_data(data_path, max_num_sample, max_num_sample_ratio):
    train_raw_data = []
    dev_raw_data = []
    if os.path.isdir(data_path):
        for f in os.listdir(data_path):
            temp_train, temp_dev = nested_load_data(os.path.join(data_path, f), max_num_sample)
            train_raw_data += temp_train
            dev_raw_data += temp_dev
        return train_raw_data, dev_raw_data
    elif os.path.isfile(data_path) and data_path.endswith('.json'):
        temp_data =  json.load(open(data_path, "r"))
        random.shuffle(temp_data)
        if max_num_sample != 0:
            temp_data = temp_data[:max_num_sample]
        elif max_num_sample_ratio != 0:
            temp_data = temp_data[:int(len(temp_data) * max_num_sample_ratio)]
        split = int(len(temp_data) * 0.98)
        train_raw_data = temp_data[:split]
        dev_raw_data = temp_data[split:]
        return train_raw_data, dev_raw_data
    else:
        return [],[]