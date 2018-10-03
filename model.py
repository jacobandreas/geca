from torchdec import hlog
from torchdec.seq import Encoder, Decoder, SimpleAttention

import numpy as np
import torch
from torch import nn

n_emb = 64
n_enc = 512
n_hid = 1024

class GeneratorModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(vocab, n_emb, n_enc, 1, bidirectional=False)
        self.decoder = Decoder(
            vocab,
            n_emb,
            n_enc,
            1,
            attention=[SimpleAttention(n_enc, n_enc)],
            copy=True,
            self_attention=True
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

    def forward(self, ctx, out):
        enc, state = self.encoder(ctx)

        copy_ctx = ctx.new_zeros(ctx.shape[0], ctx.shape[1], len(self.vocab)).float()
        for i in range(ctx.shape[0]):
            for j in range(ctx.shape[1]):
                copy_ctx[i, j, ctx[i, j]] = 1

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        att_mask = (ctx == self.vocab.pad()).float()
        pred, _ = self.decoder([enc], out_prev, state, copy_targets=[copy_ctx],
                att_mask=att_mask)
        n_batch, n_seq = out_next.shape
        pred = pred.view(n_batch * n_seq, -1)
        out_next = out_next.contiguous().view(-1)
        loss = self.loss(pred, out_next)
        return loss

    def sample(self, ctx, n_samples):
        enc, state = self.encoder(ctx)

        copy_ctx = ctx.new_zeros(ctx.shape[0], ctx.shape[1], len(self.vocab)).float()
        for i in range(ctx.shape[0]):
            for j in range(ctx.shape[1]):
                copy_ctx[i, j, ctx[i, j]] = 1

        att_mask = (ctx == self.vocab.pad()).float()

        return self.decoder.sample([enc], state, copy_targets=[copy_ctx],
                att_mask=att_mask)
