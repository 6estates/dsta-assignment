import os
import pathlib
import warnings

import pydash
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from trl import (DataCollatorForCompletionOnlyLM, ModelConfig, SFTConfig,
                 SFTTrainer, get_peft_config, get_quantization_config)

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", message=".*Could not find response key.*")
warnings.filterwarnings("ignore", message=".*You passed a model_id to the SFTTrainer.*")
warnings.filterwarnings("ignore", message=".*You passed a tokenizer with `padding_side`.*")

class JsonSerializableSFTConfig(SFTConfig):
    def to_dict(self):
        d = super().to_dict()
        q_config = pydash.get(d, 'model_init_kwargs.quantization_config')
        if q_config is not None and not isinstance(q_config, dict):
            d['model_init_kwargs']['quantization_config'] = q_config.to_dict()
        return d


def construct_device_map(use_deepspeed: bool):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if use_deepspeed:
        device_map = None
    else:
        device_map = "auto"
        if ddp:
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    return device_map


def clean_content(res):
    return (" ".join([token for token in res.split(" ") if token])).strip()

def get_pad_token_id(model_name_or_path):
    if "llama-3" in model_name_or_path.lower():
        return 128255  # '<|reserved_special_token_250|>'
    return 0  # '<unk>'

def get_resp_template(model_name_or_path):
    if "llama-3" in model_name_or_path.lower():
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return "[/INST]"


def conversations_formatting_function(tokenizer: AutoTokenizer):
    def format_dataset(examples):
        output_texts = []
        for i in range(len(examples["user_prompt"])):
            msgs = [
                {
                    "role": "system",
                    "content": "You are a financial assistant. Always provide accurate and reliable information to the best of your abilities",
                },
                {"role": "user", "content": clean_content(examples["user_prompt"][i])},
                {"role": "assistant", "content": clean_content(examples["resp"][i])},
            ]
            output_texts.append(tokenizer.apply_chat_template(msgs, tokenize=False))
        return output_texts

    return format_dataset


parser = HfArgumentParser((JsonSerializableSFTConfig, ModelConfig))
sft_args, model_config = parser.parse_args_into_dataclasses()
use_deepspeed = sft_args.deepspeed not in [None, ""]
sft_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)


################
# Model & Tokenizer
################
torch_dtype = (
    model_config.torch_dtype
    if model_config.torch_dtype in ["auto", None]
    else getattr(torch, model_config.torch_dtype)
)
quantization_config = get_quantization_config(model_config)
model_kwargs = dict(
    revision=model_config.model_revision,
    trust_remote_code=model_config.trust_remote_code,
    attn_implementation=model_config.attn_implementation,
    torch_dtype=torch_dtype,
    use_cache=False if sft_args.gradient_checkpointing else True,
    device_map=construct_device_map(use_deepspeed)
    if quantization_config is not None
    else None,
    quantization_config=quantization_config,
)
sft_args.model_init_kwargs=model_kwargs
sft_args.max_seq_length=4096
tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path, use_fast=True)
tokenizer.pad_token_id = get_pad_token_id(model_config.model_name_or_path)
tokenizer.padding_side = "left"


################
# Dataset
################
raw_datasets = load_dataset("json", data_dir=pathlib.Path(__file__).parent / "data")
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["validation"]


################
# Training
################
trainer = SFTTrainer(
    model=model_config.model_name_or_path,
    args=sft_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=conversations_formatting_function(tokenizer),
    data_collator=DataCollatorForCompletionOnlyLM(
        get_resp_template(model_config.model_name_or_path), tokenizer=tokenizer, pad_to_multiple_of=8
    ),
    tokenizer=tokenizer,
    peft_config=get_peft_config(model_config),
)
trainer.train()
trainer.save_model(sft_args.output_dir)
