from .builder import OneShotDataset

from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("cogs_dir", None, "location of cogs data")

class CogsDataset(OneShotDataset):
    def __init__(self, **kwargs):
        return super().__init__(
            self.load_split("train"),
            self.load_split("dev"),
            self.load_split("test"),
            **kwargs
        )

    def load_split(self, split):
        data = []
        with open(os.path.join(FLAGS.cogs_dir, split + ".tsv")) as reader:
            for line in reader:
                inp, out, _ = line.strip().split("\t")
                out = out.replace(" _ ", "_")
                data.append((
                    tuple(inp.split()),
                    tuple(out.split())
                ))
        return data

    def score(self, pred, ref_out, ref_inp):
        return 1 if pred == ref_out else 0
