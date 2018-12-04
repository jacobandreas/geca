import flags as _flags
from train import get_dataset

from absl import app, flags

def main(argv):
    dataset = get_dataset()
    vocab = dataset.vocab
    for inp, out in dataset.get_train():
        print(" ".join(vocab.decode(inp)))
        print(" ".join(vocab.decode(out)))
        print()

    print("\n\n\n\n")

    for inp, out in dataset.get_val():
        print(" ".join(vocab.decode(inp)))
        print(" ".join(vocab.decode(out)))
        print()

if __name__ == "__main__":
    app.run(main)
