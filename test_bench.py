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
torch.autograd.set_grad_enabled(False)
if not args.in_8bit:
    model = model.to('cuda').eval()
get_text = pipeline('text-generation', model=model, tokenizer=tokenizer)
get_sent = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
dataset = load_dataset(args.dataset, split='train')
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
sentiment_demonstrations = pd.read_csv(args.demonstrations_fp, header=None).values.tolist()
ptext=[]
psent=[]
text_sheng=[]
sent_sheng=[]
text_ours=[]
sent_ours=[]

# select the first 1000 samples in dataset
dataset = dataset.select(range(args.num_samples))
print(f"Model loaded: {model_signature}")

output_fp = f"{args.model_type}_{args.model_size}.jsonl"
for sample in dataset:
    text = sample['text']
    if sample['label'] < 3:
        ptext_=get_text(f"Please paraphrase the following text: {text} paraphrase: ", do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)[0]['generated_text']
        ptext_=ptext_.split('paraphrase: ')[1].strip()
        psent.append(get_sent(ptext_)[0]['label'])
        ptext.append(ptext_)
    else:
        ptext.append("")
        psent.append("")

icv_pos = task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))
icv_to_shift_pos = [icv_pos]
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('An ICV vector is removed\n')
    except:
        print('All ICV vectors have been removed!\n')    
        break
updated_wrapper = model_with_adapter(model)
_ = model_with_adapter(model).get_model(torch.stack(icv_to_shift_pos,dim=1).cuda(), alpha = [args.alpha])
print("Original adapter loaded")

for sample in ptext:
    if sample == "":
        text_sheng.append("")
        sent_sheng.append("")
        continue
    text_sheng_ = get_text(f"Please paraphrase the following text: {sample} paraphrase: ", do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)[0]['generated_text']
    sent_sheng.append(get_sent(text_sheng_)[0]['label'])
    text_sheng.append(text_sheng_)

icv_pos = task_agent.get_icv_ours(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))
icv_to_shift_pos = [icv_pos]
while True:
    try:
        model_with_adapter(model).remove_adapter()
        print('An ICV vector is removed\n')
    except:
        print('All ICV vectors have been removed!\n')    
        break
updated_wrapper = model_with_adapter(model)
_ = model_with_adapter(model).get_model(torch.stack(icv_to_shift_pos,dim=1).cuda(), alpha = [args.alpha])
print("New adapter loaded")

for sample in ptext:
    if sample == "":
        text_ours.append("")
        sent_ours.append("")
        continue
    text_ours_ = get_text(f"Please paraphrase the following text: {sample} paraphrase: ", do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)[0]['generated_text']
    sent_ours.append(get_sent(text_ours_)[0]['label'])
    text_ours.append(text_ours_)

dataset=dataset.add_column('ptext', ptext)
dataset=dataset.add_column('psent', psent)
dataset=dataset.add_column('text_sheng', text_sheng)
dataset=dataset.add_column('sent_sheng', sent_sheng)
dataset=dataset.add_column('text_ours', text_ours)
dataset=dataset.add_column('sent_ours', sent_ours)
dataset.save_to_disk("test.hf")
# save dataset to jsonl
with jsonlines.open(output_fp, mode='w') as writer:
    for sample in dataset:
        writer.write(sample)