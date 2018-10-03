from torchdec import hlog
from torchdec.seq import Encoder, Decoder, SimpleAttention

import numpy as np
import torch
from torch import nn, optim

n_emb = 32
n_enc = 128
n_hid = 2048

class LstmParameterizer(nn.Module):
    @profile
    def __init__(self, n_hid, n_inp, n_enc, scale):
        super().__init__()
        self.n_hid = n_hid
        self.n_inp = n_inp
        self.n_enc = n_enc
        self.scale = scale

        self.w_ih = nn.Linear(n_hid, n_inp * n_enc * 4)
        self.w_hh = nn.Linear(n_hid, n_enc * n_enc * 4)
        self.b_ih = nn.Linear(n_hid, n_enc * 4)
        self.b_hh = nn.Linear(n_hid, n_enc * 4)

        #self.w_ih = nn.Parameter(torch.Tensor(n_inp * n_enc * 4))
        #self.w_hh = nn.Parameter(torch.Tensor(n_enc * n_enc * 4))
        #self.b_ih = nn.Parameter(torch.Tensor(n_enc * 4))
        #self.b_hh = nn.Parameter(torch.Tensor(n_enc * 4))
        self._xdemo_lstm = nn.LSTM(n_emb, n_hid, 1)

        #for param in self.parameters():
        #    nn.init.uniform_(param, 1 / np.sqrt(n_enc))
        #for name, param in self.named_parameters():
        #    if "bias" in name and "_xdemo" not in name:
        #        print(param.norm())

    @profile
    def forward(self, rep):
        assert len(rep.shape) == 1

        n_hid = self.n_hid
        n_inp = self.n_inp
        n_enc = self.n_enc

        w_ih = self.w_ih(rep).view(n_enc * 4, n_inp) * self.scale
        w_hh = self.w_hh(rep).view(n_enc * 4, n_enc) * self.scale
        b_ih = self.b_ih(rep) * self.scale
        b_hh = self.b_hh(rep) * self.scale

        #for param in self._xdemo_lstm.parameters():
        #    print('r', param.shape, param.norm())

        #for param in (w_ih, w_hh, b_ih, b_hh):
        #    print('m', param.shape, param.norm())


        #w_ih = self.w_ih.view(n_enc * 4, n_inp) * self.scale
        #w_hh = self.w_hh.view(n_enc * 4, n_enc) * self.scale
        #b_ih = self.b_ih * self.scale
        #b_hh = self.b_hh * self.scale

        return (w_ih, w_hh, b_ih, b_hh)

class InductorModel(nn.Module):
    @profile
    def __init__(self, vocab):
        super().__init__()

        self.ctx_embedder = nn.Embedding(len(vocab), n_enc)
        #for param in self.ctx_embedder.parameters():
        #    nn.init.uniform_(param, 0)
        self.ctx_encoder = Encoder(vocab, n_emb, n_enc, 1, bidirectional=False)

        self.proj = nn.Conv1d(n_enc, n_hid, 1)
        self.reduce = nn.Conv1d(n_hid, n_hid, 2, 2)

        self.enc_builder = LstmParameterizer(n_hid, n_emb, n_enc, .00001)
        self.dec_builder = LstmParameterizer(n_hid, n_emb, n_enc, .00001)

        self.inp_encoder = Encoder(
            vocab, n_emb, n_enc, 1,
            bidirectional=False,
            dynamic=True
        )
        self.out_decoder = Decoder(
            vocab, n_emb, n_enc, 1, 
            attention=None,
            dynamic=True
        )

        self.loss = nn.CrossEntropyLoss()

        self.vocab = vocab

    @profile
    def forward(self, ctx, inp, out):
        # encode context
        _, (enc_ctx, _) = self.ctx_encoder(ctx)
        #enc_ctx = self.ctx_embedder(ctx).mean(dim=0).unsqueeze(0)
        #print(enc_ctx.shape)

        assert(enc_ctx.shape[0] == 1)

        # induce model
        enc_ctx = enc_ctx.permute(0, 2, 1)
        hid_ctx = self.proj(enc_ctx)
        while(hid_ctx.shape[2] > 1):
            hid_ctx = self.reduce(hid_ctx)
        hid_ctx = hid_ctx.squeeze(2).squeeze(0)

        # decode
        out_next = out[1:, :]
        out_prev = out[:-1, :]

        proc_inp, state = self.inp_encoder(inp, self.enc_builder(hid_ctx))
        #proc_inp, state = self.inp_encoder(inp)
        #print("r", [p.mean().item() for p in self.inp_encoder.parameters()])
        #dec_params = self.dec_builder(hid_ctx)
        #print("f", [p.mean().item() for p in dec_params])
        pred, _ = self.out_decoder(proc_inp, out_prev, state, self.dec_builder(hid_ctx))

        n_batch, n_seq = out_next.shape
        pred = pred.view(n_batch * n_seq, -1)
        out_next = out_next.contiguous().view(-1)
        loss = self.loss(pred, out_next)
        return loss

#class Model(nn.Module):
#    def __init__(self, vocab):
#        super().__init__()
#        self.encoder = Encoder(vocab, 32, 512, 1)
#        self.decoder = AttDecoder(vocab, 32, 1024, 512, 1)
#        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())
#
#    def _adapt(self, state):
#        out = []
#        for s in state:
#            n_batch = s.shape[1]
#            s = s.permute((1, 0, 2))
#            #s = s.contiguous().view(n_batch, -1)
#            s = s.sum(dim=1)
#            s = s.unsqueeze(0)
#            out.append(s)
#        return tuple(out)
#
#    def forward(self, inp, out):
#        ctx, state = self.encoder(inp)
#        state = self._adapt(state)
#
#        out_src = out[:, :-1]
#        out_tgt = out[:, 1:]
#        n_batch, n_seq = out_tgt.shape
#
#        pred, _ = self.decoder(ctx, out_src, state)
#        pred = pred.view(n_batch * n_seq, -1)
#        out_tgt = out_tgt.contiguous().view(n_batch * n_seq)
#        loss = self.loss(pred, out_tgt)
#        return loss
#
#    def decode(self, inp):
#        ctx, state = self.encoder(inp)
#        state = self._adapt(state)
#        return self.decoder.decode(ctx, state, DEVICE)
#
#
#vocab = Vocab()
#batch, _ = sample(10)
#for s in batch:
#    for a in s:
#        vocab.add(a)
#model = Model(vocab).to(DEVICE)
#opt = optim.Adam(model.parameters(), lr=0.001)
#
#for i in hlog.loop("%05d", range(100), timer=False):
#    def prep(ss):
#        ss = [[vocab[t] for t in s] for s in ss]
#        ss = batch_seqs(ss, vocab)
#        return ss.to(DEVICE)
#    oinp, out = sample(50)
#    inp, out = prep(oinp), prep(out)
#
#    loss = model(inp, out)
#    opt.zero_grad()
#    loss.backward()
#    opt.step()
#
#    hlog.value("loss", loss.item())
#    dec, _ = model.decode(inp[:2, :])
#    print(oinp[:2])
#    print([vocab.decode(d) for d in dec])
#    print()
