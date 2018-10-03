#!/usr/bin/env python

from grammar import GrammarFactory
from torchdec import hlog
from torchdec.seq import batch_seqs
from model import InductorModel

import numpy as np
import torch
from torch import optim
from torch.nn.utils.clip_grad import clip_grad_norm_

DEVICE = torch.device("cuda:0")
#DEVICE = torch.device("cpu")

n_ctx = 8
n_batch = 8
n_out = 8

def sample_batch(factory, vocab):
    grammar = factory()
    ctx = [vocab.encode(grammar()) for _ in range(n_ctx)]
    ctx_data = batch_seqs(ctx).to(DEVICE)

    out = [grammar(bracket=True) for _ in range(n_out)]
    #print(" ".join(out[0]))
    inp = [
        [t for t in o if not factory.is_bracket(t)]
        for o in out
    ]
    inp_data = batch_seqs([vocab.encode(i) for i in inp]).to(DEVICE)
    out_data = batch_seqs([vocab.encode(i) for i in out]).to(DEVICE)

    return ctx_data, inp_data, out_data

#forward_seqs = [
#    ["a0", "b0", "c0", "d0"],
#    ["a0", "a0", "b0", "b0"],
#    ["d0", "b0", "c0", "d0"],
#    ["b0", "d0", "a0", "d0"]
#]
#
#reverse_seqs = [list(reversed(s)) for s in forward_seqs]
#
#def sample_batch(factory, vocab):
#    reverse = np.random.randint(2)
#    #reverse = True
#    ctx = []
#    for _ in range(n_ctx):
#        i = np.random.randint(len(forward_seqs))
#        if reverse:
#            ctx.append(vocab.encode(forward_seqs[i] + reverse_seqs[i]))
#            #ctx.append(vocab.encode(["a0"]))
#        else:
#            ctx.append(vocab.encode(forward_seqs[i] + forward_seqs[i]))
#            #ctx.append(vocab.encode(["b0"]))
#
#    inp = []
#    out = []
#    for _ in range(n_out):
#        i = np.random.randint(len(forward_seqs))
#        inp.append(vocab.encode(forward_seqs[i]))
#        if reverse:
#            out.append(vocab.encode(reverse_seqs[i]))
#        else:
#            out.append(vocab.encode(forward_seqs[i]))
#
#    return batch_seqs(ctx).to(DEVICE), batch_seqs(inp).to(DEVICE), batch_seqs(out).to(DEVICE)

@profile
def main():
    factory = GrammarFactory()
    vocab = factory.vocab()
    model = InductorModel(vocab).to(DEVICE)
    opt = optim.RMSprop(model.parameters(), lr=0.0003)

    with hlog.task("train"):
        for i_epoch in hlog.loop("%05d", range(1000), timer=False):
            epoch_loss = 0
            for i_iter in range(10):
                opt.zero_grad()
                loss = 0
                for i_batch_part in range(n_batch):
                    ctx, inp, out = sample_batch(factory, vocab)
                    loss += model(ctx, inp, out)
                loss.backward()
                clip_grad_norm_(model.parameters(), .1)
                opt.step()
                epoch_loss += loss.item() / n_batch
            hlog.value("loss", epoch_loss)

if __name__ == "__main__":
    main()
