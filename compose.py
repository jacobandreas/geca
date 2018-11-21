#!/usr/bin/env python

import fakeprof

import flags as _flags
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
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

FLAGS = flags.FLAGS

flags.DEFINE_string("model", None, "name of the model to load")
flags.DEFINE_integer("n_sample", 1000, "number of training examples to sample")
flags.DEFINE_string("write", None, "path to write to")

flags.DEFINE_string("base_dir", None, "")
flags.DEFINE_string("base_model", None, "")
flags.DEFINE_string("inv_dir", None, "")
flags.DEFINE_string("inv_model", None, "")

DEVICE = torch.device("cuda:0")

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = get_dataset()

    for u in dataset.train_utts:
        print(" ".join(u[0]))
        #print(" ".join(u[1]))
    print("\n\n\n")
    for u in dataset.val_utts:
        print(" ".join(u[0]))
        #print(" ".join(u[1]))

    #model = StagedModel(
    #    dataset.vocab,
    #    copy=True,
    #    self_attention=False
    #).to(DEVICE)
    model = RetrievalModel(
        dataset.vocab
    )

    ### base_model = GeneratorModel(dataset.vocab, copy=True).to(DEVICE)
    ### base_model.load_state_dict(torch.load(os.path.join(FLAGS.base_dir, FLAGS.base_model)))
    ### inv_model = GeneratorModel(dataset.vocab, copy=True).to(DEVICE)
    ### inv_model.load_state_dict(torch.load(os.path.join(FLAGS.inv_dir, FLAGS.inv_model)))

    model.prepare(dataset)
    if isinstance(model, nn.Module):
        path = os.path.join(FLAGS.model_dir, FLAGS.model)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)

    realized = set()
    comp_data = dataset.enumerate_comp()
    while len(realized) < FLAGS.n_sample:
        try:
            templ1, templ2, names = next(comp_data)
        except StopIteration:
            break
        datum = make_batch([(templ1, templ2)], dataset.vocab, staged=True)
        (inps, outs), scores = model.sample(datum.inp_data, datum.out_data)

        ### (inp,), _ = model.sample(datum.inp_data, datum.out_data)
        ### inp = dataset.vocab.encode(dataset.realize(inp, names))
        ### print(" ".join(dataset.vocab.decode(inp)))
        ### inps = [inp for _ in range(10)]
        ### datum2 = make_batch(zip(inps, inps), dataset.vocab, staged=False)
        ### outs, scores = base_model.sample(datum2.inp_data)
        ### outs = [tuple(o) for o in outs]
        ### for out, score in set(zip(outs, scores)):
        ###     print(score, " ".join(dataset.vocab.decode(out)))
        ### print()
        ### continue

        keep = []
        for inp, out, score in zip(inps, outs, scores):
            keep.append(((inp, out), score))
        for (inp, out), score in keep:
            inp_realized = dataset.realize(inp, names)
            out_realized = dataset.realize(out, names)
            if not (
                dataset.novel(inp=inp_realized) 
                and dataset.novel(out=out_realized)
            ):
                continue
            if (inp_realized, out_realized) in realized:
                continue
        #for inp, score in zip(inps, scores):
        #    inp_realized = dataset.realize(inp, names)
        #    if not dataset.novel(inp=inp_realized):
        #        continue
        #    if inp_realized in realized:
        #        continue

            #from_inp = dataset.vocab["WUG0"] in ctx[0]
            #from_out = dataset.vocab["WUG0"] in ctx[1]
            #if (
            #    (from_inp and dataset.vocab["WUG0"] not in inp)
            #    or (from_out and dataset.vocab["WUG0"] not in out)
            #):
            #    continue
            print(score)
            print(" ".join(dataset.vocab.decode(inp)))
            print(" ".join(inp_realized))
            print("--")
            print(" ".join(dataset.vocab.decode(out)))
            print(" ".join(out_realized))
            print()
            realized.add((inp_realized, out_realized))
            #realized.add(inp_realized)

    data = [{"inp": inp, "out": out} for inp, out in realized]
    #data = [{"inp": inp} for inp in realized]
    with open(FLAGS.write, "w") as fh:
        json.dump(data, fh, indent=2)

if __name__ == "__main__":
    app.run(main)
