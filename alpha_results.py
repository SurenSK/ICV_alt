import time
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

dataset = load_dataset('json', data_files='sentiments2.jsonl', split='train')
dataset = dataset.map(lambda sample: {'length': len(sample['text'])}).sort('length')
alphas = np.linspace(0, 5, 101)
# Initialize a plot
plt.figure(figsize=(10, 6))
dataset = dataset.select([0,9])
# Iterate over samples and plot
for s in dataset:
    num_chars = len(s['text'])  # Number of characters in the sample
    sentiments = s['sentiments']  # Sentiments for this sample

    # Plotting the line for this sample
    plt.plot(alphas, sentiments, label=f"{num_chars} chars")

# Adding legend, labels, and title
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("Sentiment")
plt.title("Sentiment vs Alpha for Different Samples")

# Display the plot
plt.show()