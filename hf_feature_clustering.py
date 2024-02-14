from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
extractor = pipeline(model="tiiuae/falcon-7b", task="feature-extraction", device_map="auto")
embed = lambda inp: extractor(inp, return_tensors=True)[-1][-1]

print(embed("Hello, my dog is cute.").shape)
embeds = []
# load cat_care.txt
with open("cat_care.txt", "r") as f:
    for line in f:
        embeds.append(embed(line.strip()))
# load desk_make.txt
with open("desk_make.txt", "r") as f:
    for line in f:
        embeds.append(embed(line.strip()))
embeds = np.array(embeds)
n_samples = embeds.shape[0]
distances = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        distances[i, j] = np.linalg.norm(embeds[i] - embeds[j])

cos_sim = util.cos_sim(embeds, embeds)
import pandas as pd
df = pd.DataFrame(distances)
df.to_csv("distances.csv", index=False)