#!/usr/bin/env python

import flags as _flags
from train import get_dataset
from data.lm import LmDataset
from model import LanguageModel
from trainer import train, make_batch, Datum

from absl import app, flags, logging
import json
import numpy as np
import os
import torch
from torchdec import hlog

FLAGS = flags.FLAGS
flags.DEFINE_string("augment", None, "file with composed data for augmentation")
flags.DEFINE_float("aug_ratio", 0, "fraction of samples to draw from augmentation")
flags.DEFINE_boolean("test_curve", True, "test in place")

DEVICE = torch.device("cpu")

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
    model = LanguageModel(dataset.vocab).to(DEVICE)

    def sample():
        return dataset.sample_train(aug_ratio=FLAGS.aug_ratio)

    def callback(i_epoch):
        model.eval()
        final = i_epoch == FLAGS.n_epochs - 1
        with hlog.task("eval_val", timer=False):
            val_data = dataset.get_val()
            val_acc = evaluate(model, val_data, dataset)
        if FLAGS.TEST and (final or FLAGS.test_curve):
            with hlog.task("eval_test", timer=False):
                test_data = dataset.get_test()
                evaluate(model, test_data, dataset)
        if (i_epoch+1) % FLAGS.n_checkpoint == 0:
            torch.save(
                model.state_dict(),
                os.path.join(FLAGS.model_dir, "model.%05d.chk" % i_epoch)
            )
        return val_acc

    train(dataset, model, sample, callback, staged=False)

def evaluate(model, data, dataset):
    ppl = 0
    for i in range(0, len(data), FLAGS.n_batch):
        batch = make_batch(data[i:i+FLAGS.n_batch], model.vocab, staged=False)
        nll = model(None, batch.out_data, None, None).item()
        ppl += nll * sum(len(d) for d in data[i:i+FLAGS.n_batch])
    ppl /= sum(len(d) for d in data)
    ppl = np.exp(ppl)
    hlog.value("ppl", ppl)
    return -ppl

if __name__ == "__main__":
    app.run(main)
