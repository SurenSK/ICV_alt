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

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
from getResp import setup_llm_calls, prompt_to_sent, flush_tensors

print("Loading ICVs")
class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=1.0
    num_samples=15
    batch_size=112 #112
    in_8bit=True #True
    model_type='falcon' #falcon
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
    eos_token_id=[104,193,1001,25,1702,18858,3166]
args = Args()
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
model, tokenizer, text_pipe, sent_pipe = setup_llm_calls(args)
args.num_layers = len(model.transformer.h)
args.max_length = min(sent_pipe.tokenizer.model_max_length, text_pipe.tokenizer.model_max_length)
sentiment_demonstrations = [("Zero stars, I hate it." , "Five stars, I love it."), ("It was terrible!" , "it was awesome!"),
                            ("I would call this the worse denny's ever " , "I would call this the best denny's ever "),
                            ("I would recommend find another place." , "I would recommend this place again!"),
                            ("Would not recommend." , "Strongly recommend.")]
icv_pos_sheng = [task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]
icv_pos_ours = [task_agent.get_icv_ours(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]
print("ICVs Loaded")

dataset = load_dataset(args.dataset, split='train')
dataset = dataset.filter(lambda sample: sample['label']<3)
dataset = dataset.filter(lambda sample: len(sample['text']) < 4000)
dataset = dataset.map(lambda s: {"tokLen": len(tokenizer.encode(s["text"]))}).sort("tokLen", reverse=True)
dataset = dataset.filter(lambda s: s["tokLen"] < args.max_length)
print("Dataset Loaded")

def find_closest_indices(req_lens, tokLens):
    req_lens_np = np.array(req_lens)
    tokLens_np = np.array(tokLens)
    indices = np.array([np.abs(tokLens_np - i).argmin() for i in req_lens_np])
    return indices.tolist()
samples = dataset.select(find_closest_indices(np.linspace(0, max(dataset["tokLen"]), args.num_samples).astype(int).tolist(), dataset["tokLen"]))

print("Samples Loaded")
print(f"max tokens: {max(samples['tokLen'])} avg tokens: {sum(samples['tokLen'])/len(samples['tokLen'])}")

icvs = [icv_pos_ours] # can check other icvs later:tm:

def alpha_indicator(alpha):
    if isinstance(alpha, list):
        c0 = alpha.count(0)
        c1 = alpha.count(1)
        if c0+c1 != len(alpha):
            ret = [f"{alpha:.2f}" for alpha in alpha]
            return f"{ret}"
        if c1 == 0:
            return "None"
        if c0 == 0:
            return "All"
        type = 1 if c1 == 1 else 0
        index = alpha.index(type)
        return index if type == 1 else -index
    else:
        return f"{alpha:.2f}"

a = np.eye(args.num_layers, dtype=np.int8)
alphas = np.vstack([np.zeros(args.num_layers,dtype=np.int8),np.ones(args.num_layers,dtype=np.int8),a,a^1],dtype=np.float32).tolist()
# alphas = np.linspace(args.a0, args.a1, args.num_alphas).tolist()
# alphas = np.random.rand(100, 32).tolist()
sents = [[] for _ in samples]
confs = [[] for _ in samples]
simis = [[] for _ in samples]

print("Starting Alpha Sweep")
print(f"Total # Samples: {args.num_samples*args.num_repeats*len(alphas)*len(icvs)}")
print(f"Time Start {datetime.datetime.now()}")
report = []
for icv_num,icv in enumerate(icvs):
    for alpha_ in alphas:
        if isinstance(alpha_, list):
            alpha_ = [a*args.alpha for a in alpha_]
        t0 = time.time()
        model_with_adapter(model).set_adapter(icv, alpha_)
        resps, sents_, confs_ = prompt_to_sent(samples, args.num_repeats, text_pipe, sent_pipe)
        sents = [s + [n] for s, n in zip(sents, sents_)]
        confs = [c + [n] for c, n in zip(confs, confs_)]
        # for s,r in zip(samples["text"],resps):
        #     print(f"Sample: {s[:100]}\nResponse: {r}\n")
        resps = list(map(len,[tokenizer.encode(s) for s in resps]))
        report.append(f"ICV#{icv_num} Alpha: {alpha_indicator(alpha_)} Time: {time.time()-t0:.2f}s Positivity {sum(sents_)/len(sents_):.2f} Samples/s: {args.num_repeats*args.num_samples/(time.time()-t0):.2f}  Min/Avg/Max-RespLen: {min(resps)} {sum(resps)/len(resps):.2f} {max(resps)} Tokens/Sec: {sum(resps)/(time.time()-t0):.2f}")
        print(report[-1])
print("Alpha Sweep Complete")
for r in report:
    print(r)
samples = samples.add_column(f"sentiments", sents)
samples = samples.add_column(f"confidences", confs)
samples = samples.add_column(f"similarities", confs)
# samples.save_to_disk("sentiments")
samples.to_json("report_layers.jsonl")

df = pd.DataFrame(report, columns=["run"])
df.to_csv("report_layers.csv")
print("Done")