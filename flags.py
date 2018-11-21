from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("TEST", False, "run on test set")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_string("dataset", None, "dataset to load")
flags.DEFINE_string("model_dir", "model", "model serialization directory")
flags.DEFINE_integer("n_checkpoint", 10, "frequency with which to save")
