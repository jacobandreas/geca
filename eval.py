#!/usr/bin/env python

import flags as _flags
from train import get_dataset
from data.scan import ScanDataset
from data.copy import CopyDataset
from data.semparse import SemparseDataset
from model import GeneratorModel
from trainer import train, make_batch, Datum

from absl import app, flags, logging
import json
import numpy as np
import torch
from torchdec import hlog

FLAGS = flags.FLAGS
flags.DEFINE_string("augment", None, "file with composed data for augmentation")

DEVICE = torch.device("cuda:0")

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    hlog.flags()

    if FLAGS.augment is not None:
        with open(FLAGS.augment) as fh:
            aug_data = json.load(fh)
    else:
        aug_data = []

    dataset = get_dataset(aug_data=aug_data, dedup=FLAGS.dedup)
    model = GeneratorModel(
        dataset.vocab,
        copy=True,
        self_attention=False
    ).to(DEVICE)

    def callback(i_epoch):
        model.eval()
        with hlog.task("eval_train", timer=False):
            train_data = [dataset.sample_train() for _ in range(1000)]
            evaluate(model, train_data)
        with hlog.task("eval_val", timer=False):
            val_data = dataset.get_val()
            val_acc = evaluate(model, val_data, vis=True)
        if FLAGS.TEST:
            with hlog.task("eval_test", timer=False):
                test_data = dataset.get_test()
                evaluate(model, test_data)
        return val_acc

    train(dataset, model, dataset.sample_train, callback)

def evaluate(model, data, vis=False):
    correct = 0
    for i in range(0, len(data), FLAGS.n_batch):
        batch = make_batch(data[i:i+FLAGS.n_batch], model.vocab)
        preds, _ = model.sample(batch.inp_data, greedy=True)
        for j in range(len(preds)):
            if vis:
                logging.debug(model.vocab.decode(preds[j]))
                logging.debug(model.vocab.decode(batch.out[j]))
                logging.debug("")
            if preds[j] == batch.out[j]:
                correct += 1
    acc = 1. * correct / len(data)
    hlog.value("acc", acc)
    return acc

if __name__ == "__main__":
    app.run(main)
