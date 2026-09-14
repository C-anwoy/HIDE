# This script exists just to load models faster
import functools
import os

import torch
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          OPTForCausalLM)

from _settings import MODEL_PATH

model_dict = {'llama3-8b': "Meta-Llama-3-8B", 'llama3-8b-instruct':"Meta-Llama-3-8B-Instruct",
                "gemma2":"gemma-2-9b", "gemma2-instruct":"gemma-2-9b-it",
                "llama3-3b":"Llama-3.2-3B",
                "llama3-3b-instruct":"Llama-3.2-3B-Instruct"}

@functools.lru_cache()
def _load_pretrained_model(model_name, device, torch_dtype=torch.float16):
    if model_name.startswith('facebook/opt'):
        model = OPTForCausalLM.from_pretrained(MODEL_PATH+model_name.split("/")[1], torch_dtype=torch_dtype)
    elif model_name == "microsoft/deberta-large-mnli":
        model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")#, torch_dtype=torch_dtype)
    elif model_name in model_dict:
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_dict[model_name]), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == "falcon-7b":
        model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, trust_remote_code=True, torch_dtype=torch_dtype)
    # elif "opt" in model_name:
    #     model = AutoModelForCausalLM.from_pretrained(os.path.join(MODEL_PATH, model_name), cache_dir=None, torch_dtype=torch_dtype)
    elif model_name == 'roberta-large-mnli':
         model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")#, torch_dtype=torch_dtype)
    model.to(device)
    return model


@functools.lru_cache()
def _load_pretrained_tokenizer(model_name, use_fast=False):
    if model_name.startswith('facebook/opt-'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH+model_name.split("/")[1], use_fast=use_fast)
    elif model_name == "microsoft/deberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
    elif model_name == "roberta-large-mnli":
        tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    # elif model_name in {'llama3-3b', 'llama3-3b-instruct', 'llama3-8b', 'llama3-8b-instruct', 'mistral-7b', 'mistral-7b-instruct', 'qwen-7b', 'qwen-7b-instruct', 'qwen-14b', 'qwen-14b-instruct', 'orca2-7b','phi4-mini', 'gemma-2-9b', 'gemma-2-9b-instruct'}:
    elif model_name in model_dict:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_dict[model_name]), cache_dir=None, use_fast=use_fast)

        tokenizer.pad_token_id=tokenizer.eos_token_id
    elif model_name == "falcon-7b":
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_PATH, model_name), trust_remote_code=True, cache_dir=None, use_fast=use_fast)
    return tokenizer