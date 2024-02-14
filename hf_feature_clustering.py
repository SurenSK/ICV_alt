from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
extractor = pipeline(model="tiiuae/falcon-7b", task="feature-extraction")
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
cos_sim = util.cos_sim(embeds, embeds)
import pandas as pd

# Convert cos_sim to DataFrame
df = pd.DataFrame(cos_sim)

# Save DataFrame to CSV
df.to_csv("cos_sim.csv", index=False)
print("test")