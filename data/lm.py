from .builder import OneShotDataset

from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("lm_data_dir", None, "data directory")

TRAIN = "training.txt"
VAL = "dev.txt"
TEST = "eval.txt"

class LmDataset(OneShotDataset):
    def __init__(self, **kwargs):
        data = self.load_split(TRAIN)
        #val = self.load_split(VAL)
        #test = self.load_split(TEST)
        train = data[:10000]
        val = data[10000:-3000]
        test = data[-3000:]
        super().__init__(train, val, test, **kwargs)

    def load_split(self, split):
        data = []
        with open(os.path.join(FLAGS.lm_data_dir, split)) as f:
            for line in f:
                data.append(((), line.strip().split()))
        return data
