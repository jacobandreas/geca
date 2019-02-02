import flags as _flags
from torchdec import hlog
from torchdec.seq import Encoder, Decoder, DecoderState, SimpleAttention, batch_seqs

from absl import flags
from collections import Counter, defaultdict
import numpy as np
import torch
from torch import nn

FLAGS = flags.FLAGS
flags.DEFINE_integer("n_emb", 64, "embedding size")
flags.DEFINE_integer("n_enc", 512, "encoder hidden size")
flags.DEFINE_float("dropout", 0, "dropout probability")
flags.DEFINE_boolean("copy_sup", False, "supervised copy")
flags.DEFINE_integer("beam", None, "decode with a beam")

class RetrievalModel():
    def __init__(self, vocab):
        self.vocab = vocab

    def prepare(self, dataset):
        self.dataset = dataset

    def sample(self, inp, out):
        out = [
            s.detach().cpu().numpy().transpose().tolist()
            for s in out
        ]
        return out, [0]

class ContextModel(nn.Module):
    def __init__(self, vocab):
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
            dropout=FLAGS.dropout
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inp, out, _dout, _cout, idx):
        enc, state = self.encoder(inp)

        gather = np.zeros((1, enc.shape[1], enc.shape[2]))
        gather[...] = np.asarray(idx)[np.newaxis, :, np.newaxis]
        rep = enc.gather(0, torch.LongTensor(gather).to(_flags.device()))
        rep = self.proj(rep)
        rep = (rep, torch.zeros_like(rep))

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        n_batch, n_seq = out_next.shape
        pred, *_ = self.decoder(rep, out_prev.shape[0], out_prev)
        pred = pred.view(n_batch * n_seq, -1)
        out_next = out_next.contiguous().view(-1)
        loss = self.loss(pred, out_next)

        return loss

class StagedModel(nn.Module):
    def __init__(self, vocab, copy=False, self_attention=False):
        super().__init__()
        self.vocab = vocab
        def make_encoder():
            enc = Encoder(
                vocab,
                FLAGS.n_emb,
                FLAGS.n_enc,
                1,
                bidirectional=True,
                dropout=FLAGS.dropout,
            )
            proj = nn.Linear(FLAGS.n_enc * 2, FLAGS.n_enc)
            return enc, proj
        def make_decoder(n_att):
            return Decoder(
                vocab,
                FLAGS.n_emb,
                FLAGS.n_enc,
                1,
                attention=[
                    SimpleAttention(FLAGS.n_enc, FLAGS.n_enc) 
                    for _ in range(n_att)
                ],
                copy=copy,
                self_attention=self_attention,
                dropout=FLAGS.dropout
            )

        self.inp_encoder, self.inp_proj = make_encoder()
        self.out_encoder, self.out_proj = make_encoder()
        self.inp_decoder = make_decoder(1)
        self.out_decoder = make_decoder(2)
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

    def prepare(self, dataset):
        pass

    def forward(self, ref, tgt, _dout, _cout):
        ref_inp, ref_out = ref
        inp, out = tgt

        enc_ref_inp, state_ref_inp = self.inp_encoder(ref_inp)
        enc_inp, state_inp = self.inp_encoder(inp)
        enc_ref_out, state_ref_out = self.out_encoder(ref_out)

        enc_ref_inp = self.inp_proj(enc_ref_inp)
        state_ref_inp = [s.sum(dim=0, keepdim=True) for s in state_ref_inp]

        enc_inp = self.inp_proj(enc_inp)
        state_inp = [s.sum(dim=0, keepdim=True) for s in state_inp]

        enc_ref_out = self.out_proj(enc_ref_out)
        state_ref_out = [s.sum(dim=0, keepdim=True) for s in state_inp]

        inp_prev = inp[:-1, :]
        inp_next = inp[1:, :]
        out_prev = out[:-1, :]
        out_next = out[1:, :]

        pred_inp, *_, = self.inp_decoder(
            state_ref_inp,
            inp_prev.shape[0],
            inp_prev,
            att_features=[enc_ref_inp],
            att_tokens=[ref_inp]
        )

        pred_out, *_, = self.out_decoder(
            [state_inp[i] + state_ref_out[i] for i in range(len(state_inp))],
            out_prev.shape[0],
            out_prev,
            att_features=[enc_inp, enc_ref_out],
            att_tokens=[inp, ref_out]
        )

        n_batch, n_seq = inp_next.shape
        pred_inp = pred_inp.view(n_batch * n_seq, -1)
        inp_next = inp_next.contiguous().view(-1)
        n_batch, n_seq = out_next.shape
        pred_out = pred_out.view(n_batch * n_seq, -1)
        out_next = out_next.contiguous().view(-1)
        return self.loss(pred_inp, inp_next) + self.loss(pred_out, out_next)

    def sample(self, ref, greedy=False):
        ref_inp, ref_out = ref
        enc_ref_inp, state_ref_inp = self.inp_encoder(ref_inp)
        enc_ref_out, state_ref_out = self.out_encoder(ref_out)
        enc_ref_inp = self.inp_proj(enc_ref_inp)
        state_ref_inp = [s.sum(dim=0, keepdim=True) for s in state_ref_inp]
        enc_ref_out = self.out_proj(enc_ref_out)
        state_ref_out = [s.sum(dim=0, keepdim=True) for s in state_ref_out]

        raw_inp, inp_scores = self.inp_decoder.sample(
            state_ref_inp, 150, att_features=[enc_ref_inp], att_tokens=[ref_inp]
        )
        inp = batch_seqs(raw_inp).to(ref_inp.device)

        enc_inp, state_inp = self.inp_encoder(inp)
        enc_inp = self.inp_proj(enc_inp)
        state_inp = [s.sum(dim=0, keepdim=True) for s in state_inp]

        raw_out, out_scores = self.out_decoder.sample(
            [state_inp[i] + state_ref_out[i] for i in range(len(state_inp))],
            150, att_features=[enc_inp, enc_ref_out],
            att_tokens=[inp, ref_out]
        )
        return (raw_inp, raw_out), [si + so for si, so in zip(inp_scores, out_scores)]

class LanguageModel(nn.Module):
    def __init__(self, vocab, copy=False, self_attention=False):
        super().__init__()
        self.vocab = vocab
        self.decoder = Decoder(
            vocab,
            FLAGS.n_emb,
            FLAGS.n_enc,
            1,
            dropout=FLAGS.dropout
        )
        self.loss = nn.CrossEntropyLoss(ignore_index=vocab.pad())

    def prepare(self, dataset):
        pass

    def forward(self, inp, out, dout, cout):
        out_prev = out[:-1, :]
        out_next = out[1:, :]
        zero = torch.zeros(1, out.shape[1], FLAGS.n_enc)
        pred, *_ = self.decoder(
            (zero, zero),
            out_prev.shape[0],
            out_prev
        )
        n_batch, n_out = out_next.shape
        pred = pred.view(n_batch * n_out, -1)
        out_next = out_next.contiguous().view(-1)
        loss = self.loss(pred, out_next)
        return loss

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

    def prepare(self, dataset):
        pass

    @profile
    def forward(self, inp, out, dout, cout):
        enc, state = self.encoder(inp)
        enc = self.proj(enc)
        state = [s.sum(dim=0, keepdim=True) for s in state]

        out_prev = out[:-1, :]
        out_next = out[1:, :]
        pred, _, _, (dpred, cpred) = self.decoder(
            state,
            out_prev.shape[0],
            out_prev,
            att_features=[enc],
            att_tokens=[inp]
        )
        n_batch, n_seq = out_next.shape

        if FLAGS.copy_sup:
            dpred = torch.stack(dpred).view(n_batch * n_seq, -1)
            cpred = torch.stack(cpred).view(n_batch * n_seq, -1)
            dout_next = dout[1:, :].contiguous().view(-1)
            cout_next = cout[1:, :].contiguous().view(-1)
            loss = self.loss(dpred, dout_next) + self.loss(cpred, cout_next)
        else:
            pred = pred.view(n_batch * n_seq, -1)
            out_next = out_next.contiguous().view(-1)
            loss = self.loss(pred, out_next)

        return loss

    def sample(self, inp, greedy=False, beam=False):
        if beam and (FLAGS.beam is not None):
            preds = []
            scores = []
            for i in range(inp.shape[1]):
                p = self.beam(inp[:, i:i+1], FLAGS.beam)
                preds.append(p[0])
                scores.append(0)
            return preds, scores
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
