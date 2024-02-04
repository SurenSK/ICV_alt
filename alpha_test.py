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
from transformers.pipelines.pt_utils import KeyPairDataset
import numpy as np

class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=0.1
    num_samples=10
    batch_size=32
    truncation_len=512
    model_type='gpt2'
    model_size='sm'
    max_length=20
    num_samples = 10
    dataset_fp = "processed_dataset.jsonl"
    num_alphas = 11
    num_repeats = 3
    a0 = 0
    a1 = 3
    gpus=1
    in_8bit=False
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
setup_env(gpu_s=args.gpus, seed=args.seed)
model_signature = build_model_signature(args.model_type, args.model_size)
tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='left')
model = build_model(args.model_type, args.model_size, args.in_8bit)
if not args.in_8bit:
    model.to('cuda').eval()
text_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=args.batch_size, eos_token_id=args.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)
sent_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=args.batch_size)
TaskHandler = load_task("demo")
task_agent = TaskHandler(args.prompt_version)
task_agent.set_seed(args.seed)
sentiment_demonstrations = [("Zero stars, I hate it." , "Five stars, I love it."), ("It was terrible!" , "it was awesome!"),
                            ("I would call this the worse denny's ever " , "I would call this the best denny's ever "),
                            ("I would recommend find another place." , "I would recommend this place again!"),
                            ("Would not recommend." , "Strongly recommend.")]
icv = [task_agent.get_icv(model, tokenize_each_demonstration(tokenizer, sentiment_demonstrations))]

dataset = load_dataset(args.dataset, split='train')
dataset = dataset.map(lambda sample: {'actLength': len(sample['text'])}).sort('actLength')

alphas = np.linspace(args.a0, args.a1, args.num_alphas)
indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)
samples = dataset.select(indices)


def get_sentiment(samples):
    samples = samples.map(lambda s: {"prompt": f"paraphrase: {s['text']}"})
    samples_ = [s for s in samples["text"] for _ in range(args.num_repeats)]
    responses = [r[0]["generated_text"].split("paraphrase: ")[1] for r in text_pipe(samples_)]
    sents = [1 if s=="POSITIVE" else 0 for s in sent_pipe(responses)]
    sents = [np.mean(s) for s in np.array_split(sents, len(samples))]
    return sents

sentiments = [[] for _ in samples]
for alpha in alphas:
    while True:
        try:
            model_with_adapter(model).remove_adapter()
            print('An ICV vector is removed')
        except:
            print('All ICV vectors have been removed!')
            break
    updated_wrapper = model_with_adapter(model)
    _ = model_with_adapter(model).get_model(torch.stack(icv,dim=1).cuda(), alpha = [alpha])
    print(f'ICV added with alpha = {alpha}')

    sentiments = [s + [n] for s, n in zip(sentiments, get_sentiment(samples))]
    print(f"Measured sentiments for alpha = {alpha}")
samples = samples.add_column(f"sentiments", sentiments)
# save sentiments to file
samples.save_to_disk("sentiments")
# load sentiments from file
sentiments = Dataset.load_from_disk("sentiments")

# for each alpha-positivity curve find the alpha that gives the highest positivity with polynomial fit and basic selection
# get regression of polynomial fit coefficients to length of text and also a linear regression on basic selection
# for each sample, get the alpha that gives the highest positivity with polynomial fit and basic selection and compare with ground truth