
from models import build_model_signature, build_tokenizer, build_model
import time
from common import setup_env, mk_parser
from transformers import pipeline
import numpy as np
from datasets import load_dataset, Dataset

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
def setup_llm_calls(args):
    setup_env(gpu_s=args.gpus, seed=args.seed)
    tokenizer = build_tokenizer(args.model_type, args.model_size, padding_side='left')
    model = build_model(args.model_type, args.model_size, args.in_8bit)
    if not args.in_8bit:
        model.to('cuda').eval()
    text_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer, batch_size=args.batch_size, eos_token_id=args.eos_token_id, pad_token_id=tokenizer.pad_token_id, do_sample=True, max_new_tokens=args.max_length, top_k=args.top_k, temperature=args.temperature, num_return_sequences=1)
    sent_pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", batch_size=args.batch_size)
    return model, text_pipe, sent_pipe

def prompt_to_sent(samples, num_repeats, text_pipe, sent_pipe):
    samples = samples.map(lambda s: {"prompt": f"Please paraphrase the following text: {s['text']} paraphrase: "})
    samples_ = [s for s in samples["prompt"] for _ in range(num_repeats)]
    responses = [r[0]["generated_text"].split("paraphrase: ")[1] for r in text_pipe(samples_)]
    sents = [1 if s=="POSITIVE" else 0 for s in sent_pipe(responses)]
    sents = [np.mean(s) for s in np.array_split(sents, len(samples))]
    return sents

def flusg_tensors():
    torch.cuda.empty_cache()