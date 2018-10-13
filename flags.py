from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", None, "dataset to load")
flags.DEFINE_string("model_dir", "model", "model serialization directory")
