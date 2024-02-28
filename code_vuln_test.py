import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime
from common import setup_env, mk_parser
from models import build_model_signature, build_tokenizer, build_model
from tasks import load_task
from utils.logger import tabular_pretty_print
from utils.tools import ensure_folder
from utils.pca import PCA
from utils.llm_layers import get_layers
import numpy as np

import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import jsonlines
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyPairDataset
import random
import json

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
from getResp import setup_llm_calls, prompt_to_sent, flush_tensors
class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=1.0
    num_samples=15
    batch_size=112 #112
    in_8bit=True #True
    model_type='dolphin' #falcon
    model_size='7b' #7b
    max_new_length=20
    dataset_fp = "processed_dataset.jsonl"
    num_repeats = 11 #3
    num_alphas = 201 #101
    a0 = 0 # 0
    a1 = 4 # 5
    gpus=1
    temperature=0.45
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    seed=0
    top_k=50
args = Args()
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
model, tokenizer, text_pipe, _ = setup_llm_calls(args)
if model == 'falcon':
    args.num_layers = len(model.transformer.h)
elif model == 'dolphin':
    args.num_layers = len(model.layers)
args.max_length = text_pipe.tokenizer.model_max_length
with open('code_examples.json', 'r') as file:
    code_demos = json.load(file)
icv_code_refusal = [task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, code_demos))]

def llm_bin_classify(query):
    prompt = [{"role": "user", "content": f"Answer with yes or no only. {query}"}]
    model_inputs  = tokenizer.apply_chat_template(prompt, return_tensors="pt").to('cuda')
    generated_ids = model.generate(
                model_inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,  # Key change for getting logits
                output_scores=True              # Also required for logits
            )

    logits = generated_ids.scores[0][0]
    predicted_token_id = logits.argmax().item()
    predicted_token = tokenizer.decode(predicted_token_id)

    print(predicted_token_id, predicted_token)
    logits = generated_ids.scores[0].squeeze()
    yes_index = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_index = tokenizer.encode("no", add_special_tokens=False)[0]
    relevant_logits = torch.tensor([logits[yes_index], logits[no_index]]) 
    probabilities = torch.softmax(relevant_logits, dim=-1)

    is_vulnerable = probabilities[0] > probabilities[1]
    confidence = probabilities[0] if is_vulnerable else probabilities[1]

    return is_vulnerable.item(), confidence.item()

print(llm_bin_classify("Is the sky red?"))
print(code_demos)