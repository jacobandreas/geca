from .builder import OneShotDataset

from absl import flags
import os
import numpy as np

FLAGS = flags.FLAGS

TRAIN_DATA = [
    (["blue"], ["B"]),
    (["green"], ["G"]),
    (["red"], ["R"]),
    (["yellow"], ["Y"]),
    (["blue", "after", "green"], ["G", "B"]),
    (["red", "after", "blue"], ["B", "R"]),
    (["blue", "thrice"], ["B", "B", "B"]),
    (["red", "thrice"], ["R", "R", "R"]),
    (["blue", "around", "green"], ["B", "G", "B"]),
    (["green", "around", "red"], ["G", "R", "G"]),
    (["blue", "thrice", "after", "green"], ["G", "B", "B", "B"]),
    (["green", "after", "red", "around", "blue"], ["R", "B", "R", "G"]),
    (["blue", "after", "green", "thrice"], ["G", "G", "G", "B"]),
    (["green", "around", "red", "after", "blue"], ["B", "G", "R", "G"]),
]

TEST_DATA = [
    (["yellow", "thrice"], ["Y", "Y", "Y"]),
    (["yellow", "around", "blue"], ["Y", "B", "Y"]),
    (["red", "around", "yellow"], ["R", "Y", "R"]),
    (["yellow", "after", "red"], ["R", "Y"]),
    (["green", "after", "yellow"], ["Y", "G"]),
    (["yellow", "thrice", "after", "blue"], ["B", "Y", "Y", "Y"]),
    (["green", "after", "yellow", "thrice"], ["Y", "Y", "Y", "G"]),
    (["blue", "after", "green", "around", "yellow"], ["G", "Y", "G", "B"]),
    (["yellow", "around", "green", "after", "red", "thrice"], ["R", "R", "R", "Y", "G", "Y"]),
    (["yellow", "around", "yellow", "after", "yellow", "thrice"], ["Y", "Y", "Y", "Y", "Y", "Y"]),
]

class ColorsDataset(OneShotDataset):
    def __init__(self, **kwargs):
        super().__init__(TRAIN_DATA, TEST_DATA, TEST_DATA, **kwargs)

    def score(self, pred, ref_out, ref_inp):
        return 1 if pred == ref_out else 0
