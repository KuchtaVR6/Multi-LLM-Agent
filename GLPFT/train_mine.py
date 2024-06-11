# Usage: deepspeed train_lora.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>
import pathlib
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

from dataclasses import dataclass, field
import logging
import typing

import torch
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.utils import is_bitsandbytes_available, is_flash_attn_2_available

from train import (
    DataArguments,
    ModelArguments,
    TrainingArguments,
    make_supervised_data_module,
)

from utils.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)


# replace_llama_attn_with_flash_attn()


@dataclass
class LoraArguments:
    lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: typing.List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    if is_bitsandbytes_available():
        print("Using quantisation ..")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16)
        model_kwargs = {'quantization_config': bnb_config}
    else:
        model_kwargs = {'torch_dtype': torch.bfloat16}

    if is_flash_attn_2_available():
        print("Using flash attention ..")
        model_kwargs.update({"attn_implementation": "flash_attention_2"})

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.max_seq_length,
        padding_side="right",
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **model_kwargs
    )

    if lora_args.lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpointing:
        logging.warning(
            "gradient checkpointing with lora makes requires_grad "
            "incorrect and needs a monkey patch in Trainer or the "
            "wrapped model's forward. ref: "
            "https://github.com/lm-sys/FastChat/pull/138#issuecomment-1509172198"
        )
        model.enable_input_require_grads()

    tokenizer.pad_token = tokenizer.unk_token

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if lora_args.lora:
        data_module.update({'peft_config': lora_config})

    training_args.bf16 = True

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()