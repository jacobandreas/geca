#!/usr/bin/env python

import fakeprof

import flags as _flags
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from grammar import RgFactory
from model import GeneratorModel

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
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("n_epochs", 512, "number of training epochs")
flags.DEFINE_integer("n_epoch_batches", 32, "batches per epoch")
flags.DEFINE_integer("n_batch", 64, "batch size")
flags.DEFINE_float("lr", 0.001, "learning rate")
flags.DEFINE_float("clip", 1., "gradient clipping")

DEVICE = torch.device("cuda:0")

Datum = namedtuple("Datum", "generator ctx out ctx_data out_data")

def sample_batch(sample_fn, n):
    ctx, out = zip(*[sample_fn() for _ in range(n)])
    ctx_data = batch_seqs(ctx).to(DEVICE)
    out_data = batch_seqs(out).to(DEVICE)
    return Datum(sample_fn, ctx, out, ctx_data, out_data)

def get_dataset():
    if FLAGS.dataset == "semparse":
        return SemparseDataset()
    if FLAGS.dataset == "scan":
        return ScanDataset()
    if FLAGS.dataset == "copy":
        return CopyDataset()
    assert False, "unknown dataset %s" % FLAGS.dataset

@profile
def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    dataset = get_dataset()
    model = GeneratorModel(dataset.vocab).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr)
    #sched = opt_sched.CosineAnnealingLR(opt, T_max=None)

    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    with hlog.task("train"):
        for i_epoch in hlog.loop("%05d", range(FLAGS.n_epochs), timer=False):
            epoch_loss = 0
            for i_batch in range(FLAGS.n_epoch_batches):
                #sched.step()
                opt.zero_grad()
                datum = sample_batch(dataset.sample_train, FLAGS.n_batch)
                loss = model(datum.ctx_data, datum.out_data)
                loss.backward()
                clip_grad_norm_(model.parameters(), FLAGS.clip)
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= FLAGS.n_epoch_batches
            hlog.value("loss", epoch_loss)
            evaluate(dataset, model)
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.model_dir, "model.%05d.chk" % i_epoch)
            )

def visualize(datum, vocab, model):
    for seq in datum.ctx:
        hlog.value("inp", " ".join(vocab.decode(seq)))
    for seq in datum.out:
        hlog.value("out", " ".join(vocab.decode(seq)))
    samples = model.beam(datum.ctx_data, 10)
    for seq in samples:
        hlog.value("???", " ".join(vocab.decode(seq)))
    print()

@hlog.fn("eval", timer=False)
def evaluate(dataset, model):
    with hlog.task("train", timer=False):
        visualize(sample_batch(dataset.sample_train, 1), dataset.vocab, model)
    with hlog.task("holdout", timer=False):
        visualize(sample_batch(dataset.sample_holdout, 1), dataset.vocab, model)
    print()

if __name__ == "__main__":
    app.run(main)
