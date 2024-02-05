from getResp import setup_llm_calls, prompt_to_sent
from datasets import load_dataset
import numpy as np
class Args():
    dataset='yelp_review_full'
    demonstrations_fp="ICV_alt/sentiment_demonstrations.csv"
    alpha=1
    num_samples=10
    batch_size=32
    truncation_len=512
    model_type='gpt2'
    model_size='sm'
    max_length=20
    num_samples = 64
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
dataset = load_dataset(args.dataset, split='train')
dataset = dataset.map(lambda sample: {'actLength': len(sample['text'])}).sort('actLength')
# dataset = dataset.filter(lambda sample: sample['actLength']<2500)

alphas = np.linspace(args.a0, args.a1, args.num_alphas)
indices = np.linspace(0, len(dataset)-1, args.num_samples, dtype=int)
dataset = dataset.select(indices)

import time
text_pipe, sent_pipe = setup_llm_calls(args)
t0=time.time()
sents = prompt_to_sent(dataset, args.num_repeats, text_pipe, sent_pipe)
print(f"Time taken: {time.time()-t0} Samples/sec: {args.num_samples/(time.time()-t0)}")