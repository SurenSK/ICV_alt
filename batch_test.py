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

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import jsonlines
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=0.1
    num_samples=512
    model_type='gpt2'
    model_size='sm'
    max_length=15
    gpus=1
    in_8bit=False
    temperature=0.45
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    batch_size=32
    seed=0
    top_k=10
    eos_token_id=[104,193,1001,25,1702,18858,3166]

args = Args()
setup_env(gpu_s=args.gpus, seed=args.seed)
model_signature = build_model_signature(args.model_type, args.model_size)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='left')
model = build_model(args.model_type, args.model_size, args.in_8bit)
#model.to('cuda').eval()
dataset = load_dataset(args.dataset, split=f'train[:{args.num_samples}]')
dataset = Dataset.from_dict({"text":[f"Please paraphrase the following text {x[:500]} paraphrase: " for x in dataset["text"]]})
dataset = dataset.map(lambda e: {'length': len(e['text'])})
dataset = dataset.sort('length')

# pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

import time
report = []
for pipe_batch_size in [8, 32, 64]:
    text_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=pipe_batch_size, eos_token_id=args.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)
    for map_batch_size in [8, 32, 64]:
        dataset_=Dataset.from_dict(dataset[:64])
        t0=time.time()
        report.append("-" * 30)
        print(report[-1])
        report.append(f"Map_batch_size={map_batch_size}, Pipe_batch_size={pipe_batch_size}, num_samples={len(dataset_['text'])}")
        print(report[-1])
        dataset_ = dataset_.map(lambda sample: {"text2": [s[0]["generated_text"].split("paraphrase: ")[1] for s in text_pipe(sample["text"])]}, batched=True, batch_size=map_batch_size)
        ttot=time.time()-t0
        report.append(f"Time taken: {ttot:.2f}s, {len(dataset_['text'])/ttot:.2f} samples/s")
        print(report[-1])

    dataset_=Dataset.from_dict(dataset[:64])
    t0=time.time()
    report.append("-" * 30)
    print(report[-1])
    report.append(f"keyDataset, Pipe_batch_size={pipe_batch_size}, num_samples={len(dataset_['text'])}")
    print(report[-1])
    dataset_c = [s[0]["generated_text"].split("paraphrase: ")[1] for s in text_pipe(KeyDataset(dataset_, "text"))]
    dataset_.add_column("text2", dataset_c)
    ttot=time.time()-t0
    report.append(f"Time taken: {ttot:.2f}s, {len(dataset_['text'])/ttot:.2f} samples/s")
    print(report[-1])
        
print("FINAL REPORT")
for l in report:
    print(l)
