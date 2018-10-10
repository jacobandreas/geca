import numpy as np
from torchdec.vocab import Vocab

class CopyDataset(object):
    #chars = ["a", "b", "c", "d", "e"]
    chars = [str(i) for i in range(100)]
    def __init__(self):
        vocab = Vocab()
        for char in self.chars:
            vocab.add(char)
        self.vocab = vocab

    def _sample(self):
        seq = [np.random.choice(self.chars) for _ in range(10)]
        seq = self.vocab.encode(seq)
        return seq, seq

    def sample_train(self):
        return self._sample()

    def sample_holdout(self):
        return self._sample()

    def novel(self, utt):
        return False
