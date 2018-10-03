
from collections import Counter
import json
import numpy as np
from torchdec.vocab import Vocab
import os

DATA_DIR = "/x/jda/data/text2sql-data/data/"
DATASET = "geography.json"

min_count = 5
max_count = 15
sep = "##"
wug = "WUG"

def clean(s):
    return s.replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')

def treplace(t, old, new):
    return tuple(new if tt == old else tt for tt in t)

class SemparseDataset(object):
    def __init__(self):
        with open(os.path.join(DATA_DIR, DATASET)) as fh:
            data = json.load(fh)

        vocab = Vocab()

        dataset = {
            "train": [],
            "dev": [],
            "test": []
        }
        for query in data:
            sql = query["sql"][0]
            sql = clean(sql)
            for utt in query["sentences"]:
                built_sql = sql
                built_txt = utt["text"]
                for k, v in utt["variables"].items():
                    built_sql = built_sql.replace(k, v)
                    built_txt = built_txt.replace(k, v)

                built_sql = tuple(built_sql.split())
                built_txt = tuple(built_txt.split())
                for tok in built_sql + built_txt:
                    vocab.add(tok)
                vocab.add(sep)
                vocab.add(wug)

                dataset[utt["question-split"]].append((built_txt, built_sql))

        dataset = dataset["train"]
        unigrams = [d[0][i] for d in dataset for i in range(len(d[0]))]
        unigram_counts = Counter(unigrams)

        train_words = [
            w for w in unigrams if min_count <= unigram_counts[w] < max_count
        ]
        synth_words = [
            w for w in unigrams if unigram_counts[w] < min_count
        ]

        train_data = []
        known_utts = set()
        for w in train_words:
            usages = [(nl, mrl) for nl, mrl in dataset if w in nl]
            aug = [treplace(nl, w, wug) + (sep,) + mrl for nl, mrl in usages]
            known_utts |= set(aug)
            #aug = [(w, sep) + nl for nl, mrl in usages]
            enc = [vocab.encode(seq) for seq in aug]
            train_data.append(enc)

        synth_data = [] 
        for w in synth_words:
            usages = [(nl, mrl) for nl, mrl in dataset if w in nl]
            aug = [treplace(nl, w, wug) + (sep,) + mrl for nl, mrl in usages]
            #aug = [(w, sep) + nl for nl, mrl in usages]
            enc = [vocab.encode(seq) for seq in aug]
            synth_data += enc

        self.vocab = vocab
        self.train_data = train_data
        self.synth_data = synth_data
        self.known_utts = known_utts

    def sample(self):
        group = list(self.train_data[np.random.randint(len(self.train_data))])
        assert len(group) >= 2
        np.random.shuffle(group)
        ctx, out = group[:2]
        return ctx, out

    def novel(self, utt):
        return tuple(utt) not in self.known_utts
