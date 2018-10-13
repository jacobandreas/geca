from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_string("dataset", None, "dataset to load")
flags.DEFINE_string("model_dir", "model", "model serialization directory")
flags.DEFINE_boolean("dedup", False, "deduplicate training examples")
