#!/usr/bin/env python

import fakeprof

import flags as _flags
from model import GeneratorModel, RetrievalModel, StagedModel
from trainer import make_batch
from train import get_dataset

from absl import app, flags
from itertools import islice
import json
import numpy as np
import os
import torch
from torch import nn
from torchdec.seq import batch_seqs

MODEL_TYPES = ["retrieval", "staged"]

FLAGS = flags.FLAGS
flags.DEFINE_enum("model_type", None, MODEL_TYPES, "type of model to use")
flags.DEFINE_string("model", None, "name of the model to load")
flags.DEFINE_integer("n_sample", 1000, "number of training examples to sample")
flags.DEFINE_string("write", None, "path to write to")

DEVICE = torch.device("cuda:0")

def pick_model(dataset):
    if FLAGS.model_type == "retrieval":
        return RetrievalModel(
            dataset.vocab
        )
    elif FLAGS.model_type == "staged":
        return StagedModel(
            dataset.vocab,
            copy=True,
            self_attention=False
        ).to(DEVICE)
    elif FLAGS.model_type == "sim":
        return ContextModel(dataset.vocab)

def pick_examples(dataset):
    if FLAGS.model_type == "retrieval":
        return dataset.enumerate_comp()
    else:
        return dataset.enumerate_freq()

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = get_dataset()
    model = pick_model(dataset)

    model.prepare(dataset)
    if isinstance(model, nn.Module):
        path = os.path.join(FLAGS.model_dir, FLAGS.model)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

    realized = set()
    examples = pick_examples(dataset)
    while len(realized) < FLAGS.n_sample:
        try:
            templ, names = next(examples)
        except StopIteration:
            break
        datum = make_batch([(templ, templ) for _ in range(10)], dataset.vocab, staged=True)
        (inps, outs), scores = model.sample(datum.inp_data, datum.out_data)

        keep = []
        for inp, out, score in zip(inps, outs, scores):
            inp_realized, inp_used = dataset.realize(inp, names)
            out_realized, out_used = dataset.realize(out, names)
            if len(inp_used) == 0 or len(out_used) == 0:
                continue
            if len(inp_used | out_used) != len(names):
                continue
            if not (
                dataset.novel(inp=inp_realized) 
                and dataset.novel(out=out_realized)
            ):
                continue
            if (inp_realized, out_realized) in realized:
                continue
            keep.append(((inp_realized, out_realized), score))
        for (inp_realized, out_realized), score in keep:
            print(
                " ".join(dataset.vocab.decode(templ[0])),
                " ".join(dataset.vocab.decode(templ[1])),
                names
            )
            print(score)
            #print(" ".join(dataset.vocab.decode(inp)))
            print(" ".join(inp_realized))
            print("--")
            #print(" ".join(dataset.vocab.decode(out)))
            print(" ".join(out_realized))
            print()
            realized.add((inp_realized, out_realized))
            print(len(realized))

    data = [{"inp": inp, "out": out} for inp, out in realized]
    with open(FLAGS.write, "w") as fh:
        json.dump(data, fh, indent=2)

if __name__ == "__main__":
    app.run(main)
