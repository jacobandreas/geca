#!/usr/bin/env python

import fakeprof

import flags as _flags
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from model import GeneratorModel, RetrievalModel
from train import get_dataset

from absl import app, flags
import json
import numpy as np
import os
import torch
from torch import nn
from torchdec.seq import batch_seqs

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "name of the model to load")
flags.DEFINE_integer("n_sample", 1000, "number of training examples to sample")
flags.DEFINE_integer("wug_limit", 1, "wug limit")
flags.DEFINE_string("write", None, "path to write to")

DEVICE = torch.device("cuda:0")

n_batch = 10

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = get_dataset(dedup=FLAGS.dedup)
    #model = GeneratorModel(
    #    dataset.vocab,
    #    copy=True,
    #    self_attention=False
    #).to(DEVICE)
    model = RetrievalModel(
        dataset.vocab
    )
    model.prepare(dataset)
    if isinstance(model, nn.Module):
        path = os.path.join(FLAGS.model_dir, FLAGS.model)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

    realized = []
    while len(realized) < FLAGS.n_sample:
    #for i in range(0, FLAGS.n_sample, n_batch):
        ctx, _, names = dataset.sample_comp_gen(wug_limit=FLAGS.wug_limit)
        split = ctx.index(dataset.vocab["##"])
        c_inp, c_out = ctx[:split], ctx[split+1:]

        ctx = batch_seqs([ctx for _ in range(1)]).to(DEVICE)
        preds, scores = model.sample(ctx)

        keep = []
        for pred, score in zip(preds, scores):
            sep = dataset.vocab[dataset.sep]
            sep_indices = [i for i in range(len(pred)) if pred[i] == sep]
            if len(sep_indices) != 1:
                continue
            index, = sep_indices
            inp, out = pred[:index], pred[index+1:]
            keep.append(((inp, out), score))
        #keep = [pair for pair, score in keep]
        for (inp, out), score in keep:
            inp_realized = dataset.realize(inp, names)
            out_realized = dataset.realize(out, names)
            if not (
                dataset.novel(inp=inp_realized) 
                and dataset.novel(out=out_realized)
            ):
                continue
            from_inp = dataset.vocab["WUG0"] in c_inp
            from_out = dataset.vocab["WUG0"] in c_out
            if (
                (from_inp and dataset.vocab["WUG0"] not in inp)
                or (from_out and dataset.vocab["WUG0"] not in out)
            ):
                continue
            print(score)
            print(" ".join(dataset.vocab.decode(inp)))
            print(" ".join(inp_realized))
            print("--")
            print(" ".join(dataset.vocab.decode(out)))
            print(" ".join(out_realized))
            print()
            realized.append((inp_realized, out_realized))

    data = [{"inp": inp, "out": out} for inp, out in realized]
    with open(FLAGS.write, "w") as fh:
        json.dump(data, fh, indent=2)

if __name__ == "__main__":
    app.run(main)
