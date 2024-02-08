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

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
from getResp import setup_llm_calls, prompt_to_sent, flush_tensors

class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=1.0
    num_samples=100
    truncation_len=512
    batch_size=96 #112
    in_8bit=True #True
    model_type='falcon' #falcon
    model_size='7b' #7b
    max_length=20
    dataset_fp = "processed_dataset.jsonl"
    num_repeats = 6 #3
    num_alphas = 3 #101
    a0 = 1 # 0
    a1 = 1.6 # 5
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
args.num_layers = len(model.transformer.h)
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
samples = samples.map(lambda s: {"tokLen": len(tokenizer.encode(s["text"]))}).sort("tokLen", ascending=False)
print(f"max tokens: {max(samples['tokLen'])} avg tokens: {sum(samples['tokLen'])/len(samples['tokLen'])}")
icvs = [icv_pos_ours] # can check other icvs later:tm:
alphas = np.linspace(args.a0, args.a1, args.num_alphas)

def alpha_indicator(alpha):
    if isinstance(alpha, list):
        c0 = alpha.count(0)
        c1 = alpha.count(1)
        if c1 == 0:
            return "None"
        if c0 == 0:
            return "All"
        type = 1 if c1 == 1 else 0
        index = alpha.index(type)
        return index if type == 1 else -index
    else:
        return alpha

a=np.eye(args.num_layers, dtype=np.int8)
alphas=np.vstack([np.zeros(args.num_layers,dtype=np.int8),np.ones(args.num_layers,dtype=np.int8),a,a^1])
sents = [[] for _ in samples]
confs = [[] for _ in samples]

print("Starting Alpha Sweep")
print(f"Total # Samples: {args.num_samples*args.num_repeats*len(alphas)*len(icvs)}")
print(f"Time Start {datetime.datetime.now()}")
report = []
for icv_num,icv in enumerate(icvs):
    for alpha_ in alphas:
        alpha_adj = alpha_
        if isinstance(alpha_, list):
            alpha_adj = [a*args.alpha for a in alpha_]
        t0 = time.time()
        model_with_adapter(model).set_adapter(icv, alpha_adj)
        resps, sents_, confs_ = prompt_to_sent(samples, args.num_repeats, text_pipe, sent_pipe)
        sents = [s + [n] for s, n in zip(sents, sents_)]
        confs = [c + [n] for c, n in zip(confs, confs_)]
        # for s,r in zip(samples["text"],resps):
        #     print(f"Sample: {s[:100]}\nResponse: {r}\n")
        resps = list(map(len,[tokenizer.encode(s) for s in resps]))
        report.append(f"ICV#{icv_num} Alpha: {alpha_indicator(alpha_):.2f} Time: {time.time()-t0:.2f}s Positivity {sum(sents_)/len(sents_):.2f} Samples/s: {args.num_repeats*args.num_samples/(time.time()-t0):.2f}  Min/Avg/Max-RespLen: {min(resps)} {sum(resps)/len(resps):.2f} {max(resps)} Tokens/Sec: {sum(resps)/(time.time()-t0):.2f}")
        print(report[-1])
print("Alpha Sweep Complete")
for r in report:
    print(r)
samples = samples.add_column(f"sentiments", sents)
samples = samples.add_column(f"confidences", confs)
# samples.save_to_disk("sentiments")
samples.to_json("sentiments4_layers.jsonl")
print("Done")