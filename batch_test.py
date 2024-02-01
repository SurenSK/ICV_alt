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
class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=0.1
    num_samples=10
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


args = Args()
setup_env(gpu_s=args.gpus, seed=args.seed)
model_signature = build_model_signature(args.model_type, args.model_size)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='right')
model = build_model(args.model_type, args.model_size, args.in_8bit)
model.to('cuda').eval()
dataset = load_dataset(args.dataset, split=f'train[:{args.num_samples}]')
dataset = Dataset.from_dict({"text":[f"Please paraphrase the following text {x} paraphrase: " for x in dataset["text"]]})

get_text = pipeline('text-generation', model=model, tokenizer=tokenizer)
pipe = pipeline("text-classification", device=0)

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(get_text(dataset["text"], batch_size=batch_size, pad_token_id=get_text.tokenizer.eos_token_id, do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1), total=len(dataset["text"])):
        pass