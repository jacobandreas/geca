#!/usr/bin/env python

import fakeprof

import flags as _flags
# TODO factor out
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from model import GeneratorModel, RetrievalModel
from trainer import train, make_batch, Datum

from absl import app, flags
from collections import namedtuple
import numpy as np
import os
import torch
from torch import optim
from torch.optim import lr_scheduler as opt_sched
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchdec import hlog
from torchdec.seq import batch_seqs

FLAGS = flags.FLAGS

DEVICE = torch.device("cuda:0")

def get_dataset(**kwargs):
    if FLAGS.dataset == "semparse":
        return SemparseDataset(**kwargs)
    if FLAGS.dataset == "scan":
        return ScanDataset(**kwargs)
    if FLAGS.dataset == "copy":
        return CopyDataset(**kwargs)
    assert False, "unknown dataset %s" % FLAGS.dataset

@profile
def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    dataset = get_dataset()
    #model = GeneratorModel(
    #    dataset.vocab,
    #    copy=True,
    #    self_attention=False
    #).to(DEVICE)
    model = RetrievalModel(
        dataset.vocab
    )
    model.prepare(dataset)

    def callback(i_epoch):
        model.eval()
        evaluate(dataset, model)
        torch.save(
            model.state_dict(),
            os.path.join(FLAGS.model_dir, "model.%05d.chk" % i_epoch)
        )

    train(dataset, model, dataset.sample_comp_train, callback)

def visualize(datum, vocab, model):
    for seq in datum.inp:
        hlog.value("inp", " ".join(vocab.decode(seq)))
    for seq in datum.out:
        hlog.value("out", " ".join(vocab.decode(seq)))
    samples = model.beam(datum.inp_data, 10)
    for seq in samples:
        hlog.value("???", " ".join(vocab.decode(seq)))
    print()

@hlog.fn("eval", timer=False)
def evaluate(dataset, model):
    with hlog.task("train", timer=False):
        visualize(make_batch([dataset.sample_comp_train()], dataset.vocab), dataset.vocab, model)
    with hlog.task("holdout", timer=False):
        visualize(make_batch([dataset.sample_comp_gen()[:2]], dataset.vocab), dataset.vocab, model)
    print()

if __name__ == "__main__":
    app.run(main)
