#!/usr/bin/env python

import fakeprof

import flags as _flags
# TODO factor out
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from model import GeneratorModel, RetrievalModel, StagedModel
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

    hlog.flags()

    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)

    dataset = get_dataset()
    model = StagedModel(
        dataset.vocab,
        copy=True,
        self_attention=False
    ).to(DEVICE)
    #model = RetrievalModel(
    #    dataset.vocab
    #)
    model.prepare(dataset)

    def callback(i_epoch):
        model.eval()
        evaluate(dataset, model)
        if (i_epoch+1) % FLAGS.n_checkpoint == 0: 
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.model_dir, "model.%05d.chk" % i_epoch)
            )

    train(dataset, model, dataset.sample_comp_train, callback, staged=True)

def visualize(datum, vocab, model):
    for (inp, out) in zip(*datum.inp):
        hlog.value("inp", " ".join(vocab.decode(inp + out)))
    for (inp, out) in zip(*datum.out):
        hlog.value("out", " ".join(vocab.decode(inp + out)))
    samples, _ = model.sample(datum.inp_data, greedy=True)
    for (inp, out) in zip(*samples):
        hlog.value("???", " ".join(vocab.decode(inp + out)))
    print()

@hlog.fn("eval", timer=False)
def evaluate(dataset, model):
    with hlog.task("train", timer=False):
        visualize(
            make_batch([dataset.sample_comp_train()], dataset.vocab, staged=True), 
            dataset.vocab,
            model
        )
    #with hlog.task("holdout", timer=False):
    #    visualize(
    #        make_batch([dataset.sample_comp_gen()[:2]], dataset.vocab, staged=True), 
    #        dataset.vocab,
    #        model
    #    )
    print()

if __name__ == "__main__":
    app.run(main)
