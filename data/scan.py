from .builder import OneShotDataset

import os

DATA_DIR = "/x/jda/data/SCAN/add_prim_split"
TRAIN = "tasks_train_addprim_jump.txt"
TEST = "tasks_test_addprim_jump.txt"

class ScanDataset(OneShotDataset):
    def load_split(self, split):
        data = []
        with open(os.path.join(DATA_DIR, split)) as fh:
            for line in fh:
                toks = line.strip().split()[1:]
                split = toks.index("OUT:")
                inp = toks[:split]
                out = toks[split+1:]
                data.append((inp, out))
        return data

    def __init__(self):
        train = self.load_split(TRAIN)
        test = self.load_split(TEST)
        super().__init__(train, test, holdout={("jump",)})
