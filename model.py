from torchdec import hlog
from torchdec.seq import Encoder, Decoder, SimpleAttention

from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn

n_emb = 64
n_enc = 512

class StupidModel():
    def train(self, dataset):
        self.vocab = dataset.vocab
        counter = Counter()
        data = defaultdict(set)
        for _ in range(10000):
            ctx, out = dataset.sample_train()
            for tok in ctx:
                counter[tok] += 1
            for tok in out:
                counter[tok] += 1
            ctx, out = tuple(ctx), tuple(out)
            data[ctx].add(out)
            data[out].add(ctx)
        self.counter = counter
        self.data = data

    def generalize(self, ctx):
        c = 0
        for d in self.data.keys():
            if len(d) > 5:
                continue
            if c > 100:
                break
            print(d)
            c += 1
        print("===")
        print(ctx)
        print()

        ctx = tuple(ctx)
        if ctx in self.data:
            return self.data[ctx]
        return set()

class GeneratorModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(vocab, n_emb, n_enc, 1, bidirectional=True)
        self.proj = nn.Linear(n_enc * 2, n_enc)
        self.decoder = Decoder(
            vocab,
            n_emb,
            n_enc,
            1,
            attention=[SimpleAttention(n_enc, n_enc)],
            copy=True,
            self_attention=True,
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

    @profile
    def forward(self, ctx, out):
        enc, state = self.encoder(ctx)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        att_mask = (ctx == self.vocab.pad()).float()
        pred, _, _ = self.decoder(
            state, out_prev.shape[0], out_prev, att_features=[enc], att_tokens=[ctx]
        )
        n_batch, n_seq = out_next.shape
        pred = pred.view(n_batch * n_seq, -1)
        #print(pred.max(dim=1))
        out_next = out_next.contiguous().view(-1)
        loss = self.loss(pred, out_next)
        return loss

    def sample(self, ctx, n_samples):
        enc, state = self.encoder(ctx)
        enc = self.proj(enc)
        assert(enc.shape[1] == 1)
        enc = enc.expand(-1, n_samples, -1).contiguous()
        state = [
            s.sum(dim=0, keepdim=True).expand(-1, n_samples, -1).contiguous()
            for s in state
        ]
        return self.decoder.sample(
            state, 150, att_features=[enc], att_tokens=[ctx]
        )

    # TODO CODE DUP
    def beam(self, ctx, beam_size):
        enc, state = self.encoder(ctx)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]
        return self.decoder.beam(
            state, beam_size, 150, att_features=[enc], att_tokens=[ctx]
        )
