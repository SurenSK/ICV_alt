import gc
import json
import os
import textwrap


import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import setup_env, mk_parser
from models import build_model_signature, build_tokenizer, build_model
from tasks import load_task
from utils.logger import tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
from utils.llm_layers import get_layers
import numpy as np

def tokenize_each_demonstration(tok, demonstration_list, dataset_name=None):
    tokenized_demonstration_list = []
    for exp_id in range(len(demonstration_list)):
        demonstration_list[exp_id] = (demonstration_list[exp_id][0].strip(" .").strip("."), demonstration_list[exp_id][1].strip(" .").strip("."))

        e_original = tok(demonstration_list[exp_id][0]) 
        e_rewrite = tok(demonstration_list[exp_id][1])
        tokenized_demonstration_list.append((e_original, e_rewrite)) 
    return tokenized_demonstration_list

class AdapterLayer(torch.nn.Module):


    def __init__(self, icvs, alpha):
        super(AdapterLayer, self).__init__()
        self.icvs = icvs
        self.alpha = alpha
        self.weight_all = []

    def forward(self, x):
        input_dtype = x.dtype
        if self.icvs is not None:
            norm = torch.norm(x.float(),dim=-1).unsqueeze(-1)            
            alpha = self.alpha
            icv_all_tasks = 0
            for i in range(len(self.icvs)):
                lambda_sim = 1.0 + torch.max(torch.tensor([0.]).to(x.device), F.cosine_similarity(x.float(), self.icvs[i][None,None,:], dim=-1)).unsqueeze(-1)
                icv_all_tasks -= alpha[i] * lambda_sim * F.normalize(self.icvs[i], dim=-1).repeat(1,x.shape[1],1)
            icv_all_tasks = 0.1 * icv_all_tasks/len(self.icvs)
            
            x = F.normalize(F.normalize(x.float(),dim=-1) +  icv_all_tasks, dim=-1) * norm
            return x.type(input_dtype)
        else:
            return x

class model_with_adapter(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Freeze the original model parameters
        for params in self.model.parameters():
            params.requires_grad = False

    def get_model(self, icvs, alpha):
        for i in range(0, len(self.model.transformer.h)):
            icvs_ = icvs[i]
            self.model.transformer.h[i].mlp = torch.nn.Sequential(self.model.transformer.h[i].mlp, AdapterLayer(icvs_, alpha))
        return self.model

    def remove_adapter(self):
        
        weight_all = []
        
        for i in range(0, len(self.model.transformer.h)):
            weight_all.append(self.model.transformer.h[i].mlp[1].weight_all)
            self.model.transformer.h[i].mlp = self.model.transformer.h[i].mlp[0]
        return weight_all

class Args():
    dataset = 'demo'
    prompt_version = 'default'
    exemplar_method = 'random'
    num_k_shots = 1
    model_type = 'TinyLlama'
    model_size = '1.1B-Chat-v1.0'  # Updated to reflect TinyLlama model
    kv_iter = 15
    step_size = 0.01
    momentum = 0.9
    batch_size = 32
    gpus = 1
    seed = 0
    alpha = 1.0

args = Args()
setup_env(gpu_s=args.gpus, seed=args.seed)

# This function may need to be updated for compatibility with TinyLlama
model_signature = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
from transformers import AutoTokenizer, AutoModelForCausalLM
# Adjust tokenizer and model-building functions as needed for TinyLlama
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model.to('cuda')
torch.autograd.set_grad_enabled(False)
print(f"Model loaded: {model_signature}")

TaskHandler = load_task(args.dataset)
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
demo_safety = [("Women are inferior to men","The belief that women are inferior to men is an outdated and biased perspective rooted in historical, cultural, and social norms rather than factual evidence or inherent differences."),
        ("I think many of Jews are very money-driven and would do anything for money or personal satisfaction.","It's not right to think Jews are evil. Making generalizations about a group of people is derogatory to them and places unfair stereotypes on them."),
        ("Trans person don't deserve any help.","Every person, regardless of their gender identity or any other characteristic, deserves respect, understanding, and support."),
        ("Bitches don't even go out to have fun anymore they just go out to tweet about the function and to take pics wit other bitches.","It's not right to think women are vain.")
       ]
icv_safety = task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, demo_safety))
icvs_to_shift_safety = [icv_safety]
query_inputs_safe =  tokenizer("""This is a conversation between two people. Contex: balck people are all bitch. Response: """)

args.alpha = 1.3
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('An ICV vector is removed\n')
    except:
        print('All ICV vectors have been removed!\n')    
        break
updated_wrapper = model_with_adapter(model)
_ = updated_wrapper.get_model(torch.stack(icvs_to_shift_safety,dim=1).cuda(), alpha = [args.alpha])
print('Style vectors have been added!\n')
generation_output = model.generate(
                        input_ids=torch.tensor(query_inputs_safe['input_ids']).unsqueeze(0).cuda(),
                        attention_mask=torch.tensor(query_inputs_safe['attention_mask']).unsqueeze(0).cuda(),
                        max_new_tokens=200,
                        do_sample=True,
                        top_k=10,
                        temperature = 0.45,
                        num_return_sequences=1,
                        eos_token_id=[104,193,tokenizer.eos_token_id]
                    )
decoded_output = tokenizer.decode(generation_output[0])
print(decoded_output)