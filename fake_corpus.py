#!/usr/bin/env python

import flags as _flags
from train import get_dataset
from torchdec import hlog
import numpy as np

from absl import app, flags, logging
import json

FLAGS = flags.FLAGS

flags.DEFINE_boolean("concat", False, "concatenate with original training set")
flags.DEFINE_string("augment", None, "file with composed data for augmentation")
flags.DEFINE_string("write", None, "new corpus file")

def main(argv):
    hlog.flags()

    assert FLAGS.augment is not None
    with open(FLAGS.augment) as fh:
        aug_data = json.load(fh)

    dataset = get_dataset()
    train_data = dataset.get_train()

    utts = []
    if FLAGS.concat:
        for _, utt in train_data:
            s = " ".join(dataset.vocab.decode(utt))
            utts.append(s)
    for pair in aug_data:
        s = " ".join(pair["out"])
        utts.append(s)

    with open(FLAGS.write, "w") as fh:
        for utt in utts:
            print(utt, file=fh)

if __name__ == "__main__":
    app.run(main)
