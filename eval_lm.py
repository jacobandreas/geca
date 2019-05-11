#!/usr/bin/env python

import flags as _flags
from train import get_dataset
from data.lm import LmDataset
from model import LanguageModel
from trainer import train, make_batch, Datum

from absl import app, flags, logging
import json
import kenlm
import numpy as np
import os
import scipy.stats
import torch
from torchdec import hlog

FLAGS = flags.FLAGS
flags.DEFINE_string("augment", None, "file with composed data for augmentation")
flags.DEFINE_float("aug_ratio", 0, "fraction of samples to draw from augmentation")
flags.DEFINE_boolean("test_curve", True, "test in place")
flags.DEFINE_boolean("use_mkn", False, "train a modified Kneser--Ney LM")
flags.DEFINE_string("lm_file", None, "ARPA-format LM file")
flags.DEFINE_string("aug_lm_file", None, "ARPA-format LM file")

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    hlog.flags()

    if FLAGS.augment is not None:
        with open(FLAGS.augment) as fh:
            aug_data = json.load(fh)
    else:
        aug_data = []

    dataset = get_dataset(aug_data=aug_data)
    if FLAGS.use_mkn:
        mkn_main(dataset)
    else:
        rnn_main(dataset)

def mkn_main(dataset):
    model = kenlm.LanguageModel(FLAGS.lm_file)
    if FLAGS.aug_ratio > 0:
        assert FLAGS.aug_lm_file is not None
        aug_model = kenlm.LanguageModel(FLAGS.aug_lm_file)

    def score_utts(utts, baseline=False):
        scores = []
        for utt in utts:
            dec = " ".join(dataset.vocab.decode(utt))
            score_here = model.score(dec)
            if (not baseline) and FLAGS.aug_ratio > 0:
                #base_prob = np.exp(score_here)
                score_aug = aug_model.score(dec)

                #aug_prob = np.exp(aug_score)
                #print(np.log(base_prob), np.log(aug_prob))
                #score_here = np.log((base_prob + FLAGS.aug_ratio * aug_prob) / (1 + FLAGS.aug_ratio))
                score_here = np.logaddexp(
                    score_here + np.log(1 / (1 + FLAGS.aug_ratio)),
                    score_aug + np.log(FLAGS.aug_ratio / (1 + FLAGS.aug_ratio))
                )

            scores.append(-score_here * np.log(10))

        scores = np.asarray(scores)
        assert (scores > 0).all()
        return scores

    with hlog.task("eval_train", timer=False):
        evaluate(score_utts, dataset.get_train(), dataset)

    with hlog.task("eval_val", timer=False):
        evaluate(score_utts, dataset.get_val(), dataset)

    if FLAGS.TEST:
        with hlog.task("eval_test", timer=False):
            evaluate(score_utts, dataset.get_test(), dataset)

def rnn_main(dataset):
    model = LanguageModel(dataset.vocab).to(_flags.device())

    def sample():
        return dataset.sample_train(aug_ratio=FLAGS.aug_ratio)

    def score_utts(utts):
        fake = [((), utt) for utt in utts]
        batch = make_batch(fake, model.vocab, staged=False)
        mean = model(None, batch.out_data, None, None).item()
        tot = mean * sum(len(utt) - 1 for utt in utts)
        return tot
        
    def callback(i_epoch):
        model.eval()
        final = i_epoch == FLAGS.n_epochs - 1
        with hlog.task("eval_val", timer=False):
            val_acc = evaluate(score_utts, dataset.get_val(), dataset)
        if FLAGS.TEST and (final or FLAGS.test_curve):
            with hlog.task("eval_test", timer=False):
                evaluate(score_utts, dataset.get_test(), dataset)
        if (i_epoch+1) % FLAGS.n_checkpoint == 0:
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.model_dir, "model.%05d.chk" % i_epoch)
            )
        return val_acc

    train(dataset, model, sample, callback, staged=False)

def evaluate(score_utts, data, dataset):
    _, utts = zip(*data)
    baseline_nll = score_utts(utts, baseline=True)
    nll = score_utts(utts)
    tval, pval = scipy.stats.ttest_rel(nll, baseline_nll)
    n_toks = sum(len(utt)-1 for utt in utts)
    nll_norm = nll.sum() / n_toks
    ppl = np.exp(nll_norm)
    hlog.value("ppl", ppl)
    hlog.value("t/p", str(tval) + " " + str(pval))
    return -ppl

if __name__ == "__main__":
    app.run(main)
