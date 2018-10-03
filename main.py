#!/usr/bin/env python

from data.copy import CopyDataset
from data.semparse import SemparseDataset
from grammar import RgFactory
from model import GeneratorModel

from collections import namedtuple
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler as opt_sched
from torch.nn.utils.clip_grad import clip_grad_norm_
from torchdec import hlog
from torchdec.seq import batch_seqs

seed = 0
DEVICE = torch.device("cuda:0")

n_epochs = 500
n_epoch_batches = 10
n_batch = 64
n_val = 100
clip = 1

Datum = namedtuple("Datum", "generator ctx out ctx_data out_data")

def sample_batch(dataset, n):
    ctx, out = zip(*[dataset.sample() for _ in range(n)])
    ctx_data = batch_seqs(ctx).to(DEVICE)
    out_data = batch_seqs(out).to(DEVICE)
    return Datum(dataset, ctx, out, ctx_data, out_data)

def main():
    dataset = SemparseDataset()
    #dataset = CopyDataset()
    val_data = [sample_batch(dataset, 1) for _ in range(n_val)]
    model = GeneratorModel(dataset.vocab).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    #sched = opt_sched.CosineAnnealingLR(opt, T_max=n_epochs)

    with hlog.task("train"):
        for i_epoch in hlog.loop("%05d", range(n_epochs), timer=False):
            epoch_loss = 0
            for i_batch in range(n_epoch_batches):
                #sched.step()
                opt.zero_grad()
                datum = sample_batch(dataset, n_batch)
                loss = model(datum.ctx_data, datum.out_data)
                loss.backward()
                clip_grad_norm_(model.parameters(), clip)
                opt.step()
                epoch_loss += loss.item()
            epoch_loss /= n_epoch_batches
            hlog.value("loss", epoch_loss)
            evaluate(val_data, dataset.vocab, model)

def visualize(datum, vocab, model):
    for seq in datum.ctx:
        print(" ".join(vocab.decode(seq)))
    print("===")
    for seq in datum.out:
        print(" ".join(vocab.decode(seq)))
    print("???")
    samples, metas = model.sample(datum.ctx_data, 2)
    for seq in samples:
        print(" ".join(vocab.decode(seq)))
    for x, y in metas:
        print("%.2f" % x, "|", " ".join(["%.2f" % yy for yy in y]))
    print()


def evaluate(val_data, vocab, model):
    visualize(val_data[np.random.randint(len(val_data))], vocab, model)
    val_novel = 0
    val_ref_novel = 0
    for datum in val_data:
        samples, _ = model.sample(datum.ctx_data, 10)
        samples = [vocab.decode(s) for s in samples]
        novel = np.mean([
            datum.generator.novel(s)
            for s in samples
        ])
        ref_novel = np.mean([
            datum.generator.novel(vocab.decode(s))
            for s in datum.out
        ])
        val_novel += novel
        val_ref_novel += ref_novel
    hlog.value("novel", val_novel / len(val_data))
    hlog.value("ref_novel", val_ref_novel / len(val_data))

if __name__ == "__main__":
    torch.manual_seed(seed)
    np.random.seed(seed)
    main()
