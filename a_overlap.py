#!/usr/bin/env python3

from train import get_dataset

from absl import app
import json

AUG = "exp/scan_jump/retrieval/composed.%d.json"

def main(argv):
    for i in range(10):
        with open(AUG % i) as fh:
            aug_data = json.load(fh)
        dataset = get_dataset(aug_data = aug_data)

if __name__ == "__main__":
    app.run(main)
