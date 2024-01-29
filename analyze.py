import pandas as pd
import json

class fscore():
    def __init__(self, name, tp=0, fp=0, fn=0, tn=0):
        self.name=name
        self.tp=tp
        self.fp=fp
        self.fn=fn
        self.tn=tn

    @property
    def recall(self):
        return 0.0 if self.tp+self.fn == 0 else self.tp / (self.tp + self.fn)
    @property
    def precision(self):
        return 0.0 if self.tp+self.fp == 0 else self.tp / (self.tp + self.fp)
    @property
    def f1_score(self):
        return 0.0 if self.precision+self.recall == 0 else 2 * self.precision * self.recall / (self.precision + self.recall)
    
    def __str__(self):
        return f"{self.name}: "\
            f"True Positives: {self.tp}, False Positives: {self.fp}, "\
            f"False Negatives: {self.fn}, True Negatives: {self.tn}, "\
            f"F1 Score: {round(self.f1_score,2)}"
    def __truediv__(self, other):
        d = lambda n, d: round(n / d, 2) if d != 0 else float('inf')
        return f"{self.name}/{other.name}, True Positives: {d(self.tp, other.tp)}, "\
            f"False Positives: {d(self.fp, other.fp)}, False Negatives: {d(self.fn, other.fn)}, "\
            f"True Negatives: {d(self.tn, other.tn)}"
        

def analyze_fp(fp):
    dataset = pd.read_json(fp, lines=True)
    ours = fscore("ours")
    icv = fscore("icv")

    for index, sample in dataset.iterrows():
        a, o, s = sample['psent'], sample['sent_ours'], sample['sent_sheng']
        if a == "POSITIVE":
            if o == "NEGATIVE":
                ours.tn += 1
            else:
                ours.fp += 1
            if s == "NEGATIVE":
                icv.tn += 1
            else:
                icv.fp += 1
        else:
            if o == "POSITIVE":
                ours.tp += 1
            else:
                ours.fn += 1
            if s == "POSITIVE":
                icv.tp += 1
            else:
                icv.fn += 1

    print(ours)
    print(icv)
    print(ours/icv)