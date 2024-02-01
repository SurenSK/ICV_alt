from transformers import pipeline
from datasets import load_dataset
import datasets
from tqdm.auto import tqdm
# from torch.utils.data import Dataset
from datasets import Dataset

from time import sleep
datasets.logging.set_verbosity_info()
dataset = load_dataset('yelp_review_full', split='train[:1%]')
dataset = Dataset.from_dict({"text":[x[:200] for x in dataset["text"]]})
pipe = pipeline("text-classification", device=0)

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"Streaming batch_size={batch_size}")
    for out in tqdm(pipe(dataset["text"], batch_size=batch_size), total=len(dataset["text"])):
        pass