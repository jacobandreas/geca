from .builder import OneShotDataset

from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string("scan_data_dir", None, "data directory")
flags.DEFINE_string("scan_split", "add_prim_split", "data split")
flags.DEFINE_string("scan_file", "addprim_jump", "data file")

TRAIN = "tasks_train_%s.txt"
TEST = "tasks_test_%s.txt"
REF = "../tasks.txt"

class ScanDataset(OneShotDataset):
    def __init__(self, **kwargs):
        train = self.load_split(TRAIN % FLAGS.scan_file)
        np.random.shuffle(train)
        val = train[:FLAGS.n_batch * 10]
        train = train[FLAGS.n_batch * 10:]
        if FLAGS.TEST:
            test = self.load_split(TEST % FLAGS.scan_file)
        else:
            test = val

        ref_data = self.load_split(REF)
        self.ref = {tuple(k): tuple(v) for k, v in ref_data}

        super().__init__(
            train, val, test, 
            #holdout={("jump",), ("I_JUMP",)},
            **kwargs
        )

    def load_split(self, split_file):
        data = []
        with open(os.path.join(FLAGS.scan_data_dir, FLAGS.scan_split, split_file)) as fh:
            for line in fh:
                toks = line.strip().split()[1:]
                split = toks.index("OUT:")
                inp = toks[:split]
                out = toks[split+1:]
                data.append((inp, out))
        return data

    def score(self, pred, ref_out, ref_inp):
        if FLAGS.invert:
            # NACS eval
            pred_str = tuple(self.vocab.decode(pred))
            inp_str = tuple(self.vocab.decode(ref_inp))
            if pred_str not in self.ref:
                return 0
            return 1 if self.ref[pred_str] == inp_str else 0
        else:
            return 1 if pred == ref_out else 0
