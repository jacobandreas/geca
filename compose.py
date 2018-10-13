#!/usr/bin/env python

import fakeprof

import flags as _flags
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from model import GeneratorModel
from train import get_dataset

from absl import app, flags
import json
import os
import torch
from torchdec.seq import batch_seqs

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "name of the model to load")
flags.DEFINE_integer("n_sample", 1000, "number of training examples to sample")
flags.DEFINE_integer("wug_limit", 1, "wug limit")
flags.DEFINE_string("write", None, "path to write to")

DEVICE = torch.device("cuda:0")

n_batch = 10

def main(argv):
    dataset = get_dataset()
    model = GeneratorModel(dataset.vocab).to(DEVICE)
    path = os.path.join(FLAGS.model_dir, FLAGS.model)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)

    realized = []
    for i in range(0, FLAGS.n_sample, n_batch):
        ctx, _, names = dataset.sample_holdout(wug_limit=FLAGS.wug_limit)
        ctx = batch_seqs([ctx]).to(DEVICE)
        generate = min(FLAGS.n_sample - i, n_batch)
        print(generate)
        preds, scores = model.sample(ctx, generate)

        keep = []
        for pred, score in zip(preds, scores):
            sep = dataset.vocab[dataset.sep]
            sep_indices = [i for i in range(len(pred)) if pred[i] == sep]
            if len(sep_indices) != 1:
                continue
            index, = sep_indices
            inp, out = pred[:index], pred[index+1:]
            keep.append(((inp, out), score))
        keep = [pair for pair, score in keep]
        batch_realized = [
            (dataset.realize(inp, names), dataset.realize(out, names))
            for inp, out in keep
        ]
        realized += batch_realized

    data = [{"inp": inp, "out": out} for inp, out in realized]
    with open(FLAGS.write, "w") as fh:
        json.dump(data, fh, indent=2)

if __name__ == "__main__":
    app.run(main)
