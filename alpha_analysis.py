import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
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

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
from getResp import setup_llm_calls, prompt_to_sent, flush_tensors

class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=1.0
    num_samples=10
    batch_size=112 #112
    truncation_len=512
    in_8bit=True #True
    model_type='falcon' #falcon
    model_size='7b' #7b
    max_length=20
    dataset_fp = "processed_dataset.jsonl"
    num_repeats = 3 #3
    num_alphas = 101 #101
    a0 = 0 # 0
    a1 = 5 # 3
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
    eos_token_id=[104,193,1001,25,1702,18858,3166]
args = Args()
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
model, tokenizer, text_pipe, sent_pipe = setup_llm_calls(args)
sentiment_demonstrations = [("Zero stars, I hate it." , "Five stars, I love it."), ("It was terrible!" , "it was awesome!"),
                            ("I would call this the worse denny's ever " , "I would call this the best denny's ever "),
                            ("I would recommend find another place." , "I would recommend this place again!"),
                            ("Would not recommend." , "Strongly recommend.")]
icv_pos_sheng = [task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]
icv_pos_ours = [task_agent.get_icv_ours(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]

dataset = load_dataset(args.dataset, split='train')
dataset = dataset.filter(lambda sample: sample['label']<3)
dataset = dataset.filter(lambda sample: len(sample['text']) < 2500)
indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)

samples = dataset.select(indices)
icvs = [icv_pos_ours] # can check other icvs later:tm:
alphas = np.linspace(args.a0, args.a1, args.num_alphas)
sents = [[] for _ in samples]

print("Starting Alpha Sweep")
print(f"Total # Samples: {args.num_samples*args.num_repeats*args.num_alphas*len(icvs)}")

for icv_num,icv in enumerate(icvs):
    for alpha_ in alphas:
        t0 = time.time()
        model_with_adapter(model).set_adapter(icv, alpha_)
        resps, sents_ = prompt_to_sent(samples, args.num_repeats, text_pipe, sent_pipe)
        sents = [s + [n] for s, n in zip(sents, sents_)]
        resps = tokenizer.encode_batch(resps)
        maxLen, totLen = max(map(len, resps)), sum(map(len, resps))
        print(f"ICV#{icv_num} Alpha: {alpha_:.2f} Time: {time.time()-t0:.2f}s Samples/s: {args.num_repeats*args.num_samples/(time.time()-t0):.2f} Max Len Resp: {maxLen} Tokens/Sec: {totLen/(time.time()-t0):.2f}")
samples = samples.add_column(f"sentiments", sents)
# samples.save_to_disk("sentiments")
samples.to_json("sentiments2.jsonl")
print("Done")