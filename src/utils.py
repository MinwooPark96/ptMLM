import os
import regex as re
import logging
import torch.nn as nn
import json
import torch 
import random
import numpy as np
import inspect
from typing import Union, Iterable, Optional
import sys
sys.path.append('..')

logger = logging.getLogger(__name__)


def count_trainable_parameters(model):
    """[minwoo] source from : https://github.com/ZhengxiangShi/PowerfulPromptFT/blob/main/src/model.py#L243"""
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_trainable_params}/{num_params}")
    return num_trainable_params

def random_mask_input_ids(input_ids, mask_token_id, exceptions, prob=0.15):
    # generate randomly masked input_ids for MLM task
    # 현재 task 에선 사용할 필요 없을 듯.
    # modified from https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
    """
    [minwoo] source from : https://github.com/salesforce/Overture/blob/main/utils.py
    exceptions: list, token ids that should not be masked
    """
    probs = torch.rand(input_ids.shape)
    mask = probs < prob
    for ex_id in exceptions:
        mask = mask * (input_ids != ex_id)
    selection = []
    for i in range(input_ids.shape[0]):
        selection.append(torch.flatten(mask[i].nonzero()).tolist())
    for i in range(input_ids.shape[0]):
        input_ids[i, selection[i]] = mask_token_id
    return input_ids

def set_seed(seed=0):
    """[minwoo] source from : https://github.com/salesforce/Overture/blob/main/utils.py"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return True

def wrap(v_or_vs: Union[str, Iterable[str]]) -> Optional[frozenset[str]]:
    """[minwoo] wrap the input into frozenset if it is not None."""
    if v_or_vs is None:
        return None
    if isinstance(v_or_vs, str):
        return frozenset({v_or_vs})
    else:
        return frozenset(v_or_vs)

def freeze_params_except(model: nn.Module, except_keyword: str):
    """[minwoo] freeze all params except the ones with except_keyword in their name"""
    for n,p in model.named_parameters():
        if except_keyword not in n:
            p.requires_grad = False

def print_params_only_requires_grad_true(model: torch.nn.Module):
    """[minwoo] print all params with requires_grad=True"""
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n, p.requires_grad)

def print_params_with_requires_grad(model: torch.nn.Module):
    """[minwoo] print all params"""
    for n,p in model.named_parameters():
        print(n, p.requires_grad)

def get_params_dict(model: torch.nn.Module):
    """[minwoo] return params from torch model"""
    return {n:p for n,p in model.named_parameters()}

def print_object_methods(obj):
    """[minwoo] print all methods of the object"""
    methods = inspect.getmembers(obj, predicate=inspect.ismethod)
    for name, method in methods:
        print(name)

def print_object_fields(obj):
    """[minwoo] print all fields of the object"""
    fields = vars(obj)
    for field_name, field_value in fields.items():
        print(f"{field_name}: {field_value}")

def create_dir(output_dir):
    """
    Checks whether to the output_dir already exists and creates it if not.
    Args:
      output_dir: path to the output_dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for p in model.parameters():
        p.requires_grad = False
    
# def get_last_checkpoint(output_dir):
#     if os.path.exists(os.path.join(output_dir, 'pytorch_model.bin')):
#         return output_dir
#     return None

def get_checkpoint(output_dir: str,
                   resume_from_checkpoint:Optional[str] = None) -> Optional[str]:
    """[minwoo] return the checkpoint path if it exists in the output_dir. Otherwise, return None."""
    if file_list := os.listdir(output_dir):
        if resume_from_checkpoint in file_list:
            return os.path.join(output_dir, resume_from_checkpoint)
        checkpoint_list = sorted([checkpoint for checkpoint in file_list if checkpoint.endswith('bin')])
        return os.path.join(output_dir, checkpoint_list[-1])
    
    return None

#[minwoo] TODO CHECK!!
def pad_punctuation(text):
    raise NotImplementedError
    """Re-implementation of _pad_punctuation in t5. This function adds spaces
    around punctuation. While this pads punctuation as expected, it has the 
    unexpected effected of padding certain unicode characters with accents, with
    spaces as well. For instance: "François" becomes "Fran ç ois"""
    # Pad everything except for: underscores (_), whitespace (\s),
    # numbers (\p{N}), letters (\p{L}) and accent characters (\p{M}).
    text = re.sub(r'([^_\s\p{N}\p{L}\p{M}])', r' \1 ', str(text))
    # Collapse consecutive whitespace into one space.
    text = re.sub(r'\s+', ' ', text)
    return text

def save_json(filepath, dictionary):
    with open(filepath, "w") as outfile:
        json.dump(dictionary, outfile)

def read_json(filepath):
    f = open(filepath,)
    return json.load(f)

def save_training_config(config_file, output_dir):
    json_data = read_json(config_file)
    save_json(os.path.join(output_dir, "training_config.json"), json_data)
            
def prepend_virtual_tokens(inputs: dict[str,torch.Tensor],
                          soft_prompt_length: int) -> None:
    
    """[minwoo] prepend virtual tokens to the input_ids and attention_mask"""
    batch_size = inputs['input_ids'].size(0)
    virtual_tokens = torch.tensor([-i for i in range(soft_prompt_length, 0,-1)]).repeat(batch_size,1)
    
    inputs['input_ids'] = torch.cat([virtual_tokens, inputs['input_ids']], 1)
    inputs['attention_mask'] = torch.cat([torch.full((batch_size,soft_prompt_length), 1), inputs['attention_mask']], 1)
    

if __name__ == '__main__':
    
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(["May the force be", "hello, my name is dog"],
                        padding=True, truncation=True,
                       return_tensors="pt")

    prepend_virtual_tokens(inputs, 100)
    print(inputs['input_ids'])  
