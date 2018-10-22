from .builder import OneShotDataset

from absl import flags
from collections import Counter
import json
import numpy as np
from torchdec.vocab import Vocab
import os

FLAGS = flags.FLAGS
flags.DEFINE_string("semparse_split", "question", "which split of the semparse dataset to use")

DATA_DIR = "/x/jda/data/text2sql-data/data/"
DATASET = "geography.json"

def clean(s):
    return s.replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')

class SemparseDataset(OneShotDataset):
    def __init__(self, **kwargs):
        with open(os.path.join(DATA_DIR, DATASET)) as fh:
            data = json.load(fh)

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

                if FLAGS.semparse_split == "question":
                    split = utt["question-split"]
                elif FLAGS.semparse_split == "query":
                    split = query["query-split"]
                else:
                    assert False, "unknown split %s" % FLAGS.split

                dataset[split].append((built_txt, built_sql))

        if FLAGS.TEST:
            dataset["train"] += dataset["dev"]

        super().__init__(
            dataset["train"],
            dataset["dev"],
            dataset["test"],
            **kwargs
        )
