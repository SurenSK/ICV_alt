import time
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset('json', data_files='sentiments4_layers.jsonl', split='train')
alphas = np.linspace(0, 5, 201)
# Initialize a plot
plt.figure(figsize=(10, 6))
# dataset = dataset.select([0,9])
# Iterate over samples and plot
for s in dataset:
    num_toks = s['tokLen']  # Number of tokens in this sample
    sentiments = s['sentiments']  # Sentiments for this sample
    acceptable_alphas = [alpha for alpha, sentiment in zip(alphas, sentiments) if sentiment > 0.9]
    acceptable_alpha = sum(acceptable_alphas)/len(acceptable_alphas)
    print(f"Toks: {num_toks} Avgλ: {acceptable_alpha:.2f} Minλ: {min(acceptable_alphas):.2f} Maxλ: {max(acceptable_alphas):.2f}")
    # Plotting the line for this sample
    plt.plot(alphas, sentiments, label=f"{num_toks} chars, λ: {acceptable_alpha:.2f}")

# Adding legend, labels, and title
plt.legend()
plt.xlabel("Layer")
plt.ylabel("Sentiment")
plt.title("Sentiment vs Layer# for Different Samples")

# Display the plot
plt.show()