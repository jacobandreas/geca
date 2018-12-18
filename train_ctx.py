import fakeprof

import flags as _flags
from data.semparse import SemparseDataset
from model import ContextModel
from train import get_dataset
from trainer import train

from absl import app, flags
import numpy as np
import os
import torch
from torchdec import hlog

FLAGS = flags.FLAGS

def main(argv):
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    hlog.flags()
    if not os.path.exists(FLAGS.model_dir):
        os.mkdir(FLAGS.model_dir)
    dataset = get_dataset()
    model = ContextModel(dataset.vocab).to(_flags.device())
    def callback(i_epoch):
        pass
    train(dataset, model, dataset.sample_ctx_train, callback, staged=False)

if __name__ == "__main__":
    app.run(main)
