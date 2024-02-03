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

from model_with_adapter import tokenize_each_demonstration, AdapterLayer, model_with_adapter
import pandas as pd
from transformers import pipeline
from datasets import load_dataset
import jsonlines
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

class Args():
    dataset='yelp_review_full'
    demonstrations_fp="sentiment_demonstrations.csv"
    alpha=1.0
    num_samples=512
    batch_size=32
    truncation_len=512
    model_type='falcon'
    model_size='7b'
    max_length=20
    gpus=1
    in_8bit=True
    temperature=0.7
    prompt_version='default'
    exemplar_method='random'
    num_k_shots=1
    kv_iter= 15
    step_size=0.01
    momentum=0.9
    batch_size=32
    seed=0
    top_k=50

args = Args()
setup_env(gpu_s=args.gpus, seed=args.seed)
model_signature = build_model_signature(args.model_type, args.model_size)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='left')
model = build_model(args.model_type, args.model_size, args.in_8bit)
if not args.in_8bit:
    model.to('cuda').eval()
#text_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=args.batch_size, pad_token_id=tokenizer.eos_token_id, do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1, top_p=0.75, eos_token_id=[104,193,1001,25,1702,18858,3166])
text_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=args.batch_size, max_new_tokens=15,
                                             do_sample=True,
                                             temperature=0.7,
                                             top_p=0.75,
                                             top_k=50,
                                             eos_token_id=[104,193,1001,25,1702,18858,3166],)
sent_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=args.batch_size, device=0)
def get_prompt(samples):
    output=[]
    for sample in samples:
        output.append(f"Please paraphrase the following text: {sample} paraphrase: ")
    return output

print("Forming ICVs")
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
sentiment_demonstrations = [["Zero stars, I hate it.", "Five stars, I love it."],
["It was terrible!", "it was awesome!"],
["I would call this the worse denny's ever ", "I would call this the best denny's ever "],
["I would recommend find another place.", "I would recommend this place again!"],
["Would not recommend.", "Strongly recommend."]]
icv_pos_sheng = [task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]
icv_pos_ours = [task_agent.get_icv_ours(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]
print("Formed ICVs")

print("Loading dataset")
dataset = load_dataset(args.dataset, split='train')
dataset = dataset.filter(lambda sample: sample['label']<3).select(range(args.num_samples))
print(f"Finished loading dataset, number of samples: {len(dataset)}\n")

print("Preprocessing dataset")
t0 = time.time()
dataset = dataset.map(lambda sample: {'trText': sample['text'][:args.truncation_len]})
dataset = dataset.map(lambda sample: {'length': len(sample['trText'])}).sort('length')
dataset = dataset.map(lambda sample: {"trSent": [s["label"] for s in sent_pipe(sample["trText"])]}, batched=True, batch_size=8)
dataset = dataset.map(lambda sample: {"trPrompt": get_prompt(sample["text"])}, batched=True, batch_size=1000)
print(f"Finished preprocessing dataset, time: {time.time()-t0} seconds\n")

query_inputs_sentiment =  tokenizer("""Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: """)


#for i in range(20):
#    print(text_pipe("Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: "))


print("Processing no ICV")
dataset = dataset.map(lambda sample: {"nText": [s[0]["generated_text"].split('paraphrase: ')[1].strip() for s in text_pipe(sample["trPrompt"])]}, batched=True)
dataset = dataset.map(lambda sample: {"nSent": [s["label"] for s in sent_pipe(sample["nText"])]}, batched=True)
print("Finished processing no ICV")

print("Processing Sheng ICV")
t0 = time.time()
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('An ICV vector is removed')
    except:
        print('All ICV vectors have been removed!')    
        break
updated_wrapper = model_with_adapter(model)
_ = updated_wrapper.get_model(torch.stack(icv_pos_sheng,dim=1).cuda(), alpha = [args.alpha])

#for i in range(20):
#    print(text_pipe("Please paraphrase the following sentence. Sentence: Worst restaurant ever!, paraphrase: "))
#exit()

dataset = dataset.map(lambda sample: {"shengText": [s[0]["generated_text"].split('paraphrase: ')[1].strip() for s in text_pipe(sample["trPrompt"])]}, batched=True)
dataset = dataset.map(lambda sample: {"shengSent": [s["label"] for s in sent_pipe(sample["shengText"])]}, batched=True)
print(f"Finished processing Sheng ICVs, time: {time.time()-t0} seconds\n")

print("Processing our ICV")
t0 = time.time()
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('An ICV vector is removed')
    except:
        print('All ICV vectors have been removed!')    
        break
updated_wrapper = model_with_adapter(model)
_ = model_with_adapter(model).get_model(torch.stack(icv_pos_ours,dim=1).cuda(), alpha = [args.alpha])
dataset = dataset.map(lambda sample: {"ourText": [s[0]["generated_text"].split('paraphrase: ')[1].strip() for s in text_pipe(sample["trPrompt"])]}, batched=True)
dataset = dataset.map(lambda sample: {"ourSent": [s["label"] for s in sent_pipe(sample["ourText"])]}, batched=True)
print(f"Finished processing Our ICVs, time: {time.time()-t0} seconds\n")

a,n,o,s=0,0,0,0
tot=len(dataset)
for sample in dataset:
    a+=1 if sample["trSent"]=="POSITIVE" else 0
    n+=1 if sample["nSent"]=="POSITIVE" else 0
    s+=1 if sample["shengSent"]=="POSITIVE" else 0
    o+=1 if sample["ourSent"]=="POSITIVE" else 0
print(f"Positivity - Base: {a/tot}, no ICV: {n/tot}, Sheng: {s/tot}, Ours: {o/tot}")

dataset.to_json("whatever.jsonl")
