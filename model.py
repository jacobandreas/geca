from torchdec import hlog
from torchdec.seq import Encoder, Decoder, SimpleAttention

from absl import flags
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_emb", 64, "embedding size")
flags.DEFINE_integer("n_enc", 512, "encoder hidden size")
flags.DEFINE_float("dropout", 0, "dropout probability")

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
    def __init__(self, vocab, copy=False, self_attention=False):
        super().__init__()
        self.vocab = vocab
        self.encoder = Encoder(
            vocab,
            FLAGS.n_emb,
            FLAGS.n_enc,
            1,
            bidirectional=True,
            dropout=FLAGS.dropout
        )
        self.proj = nn.Linear(FLAGS.n_enc * 2, FLAGS.n_enc)
        self.decoder = Decoder(
            vocab,
            FLAGS.n_emb,
            FLAGS.n_enc,
            1,
            attention=[SimpleAttention(FLAGS.n_enc, FLAGS.n_enc)],
            copy=copy,
            self_attention=self_attention,
            dropout=FLAGS.dropout
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

    @profile
    def forward(self, inp, out, dout, cout):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        att_mask = (inp == self.vocab.pad()).float()
        pred, _, _, (dpred, cpred) = self.decoder(
            state,
            out_prev.shape[0],
            out_prev,
            att_features=[enc],
            att_tokens=[inp]
        )
        n_batch, n_seq = out_next.shape

        #pred = pred.view(n_batch * n_seq, -1)
        #out_next = out_next.contiguous().view(-1)
        #loss = self.loss(pred, out_next)
        #return loss

        #pred = pred.view(n_batch * n_seq, -1)
        dpred = torch.stack(dpred).view(n_batch * n_seq, -1)
        cpred = torch.stack(cpred).view(n_batch * n_seq, -1)
        dout_next = dout[1:, :].contiguous().view(-1)
        cout_next = cout[1:, :].contiguous().view(-1)
        loss = self.loss(dpred, dout_next) + self.loss(cpred, cout_next)
        return loss

    def sample(self, inp, greedy=False):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        #assert(enc.shape[1] == 1)
        #enc = enc.expand(-1, n_samples, -1).contiguous()
        state = [
            #s.sum(dim=0, keepdim=True).expand(-1, n_samples, -1).contiguous()
            s.sum(dim=0, keepdim=True)
            for s in state
        ]
        return self.decoder.sample(
            state, 150, att_features=[enc], att_tokens=[inp], greedy=greedy
        )

    # TODO CODE DUP
    def beam(self, inp, beam_size):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]
        return self.decoder.beam(
            state, beam_size, 150, att_features=[enc], att_tokens=[inp]
        )
