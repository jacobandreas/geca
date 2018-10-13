from collections import defaultdict
import numpy as np
from torchdec.vocab import Vocab

max_wugs = 3
wug_template = "WUG%d"
sep = "##"

def t_subseq(subseq, seq):
    for i in range(len(seq)-len(subseq)+1):
        if seq[i:i+len(subseq)] == subseq:
            return True
    return False

def t_replace(old_subseq, new_subseq, seq):
    for i in range(len(seq)-len(old_subseq)+1):
        if seq[i:i+len(old_subseq)] == old_subseq:
            return seq[:i] + new_subseq + seq[i+len(old_subseq):]
    return seq

def t_replace_all(old_subseq, new_subseq, seq):
    assert not t_subseq(old_subseq, new_subseq)
    before = None
    after = seq
    while after != before:
        before = after
        after = t_replace(old_subseq, new_subseq, before)
    return after

class OneShotDataset(object):
    def __init__(self, train_utts, test_utts, ngram_size=1, holdout=frozenset()):
        index = defaultdict(set)
        vocab = Vocab()
        for i_wug in range(max_wugs):
            vocab.add(wug_template % i_wug)
        vocab.add(sep)
        for inp, out in train_utts:
            for i in range(len(inp)-ngram_size+1):
                ngram = tuple(inp[i:i+ngram_size])
                index[ngram].add((tuple(inp), tuple(out)))
                for word in inp + out:
                    vocab.add(word)

        self.vocab = vocab

        keys = sorted(list(index.keys()))
        counts = np.asarray([len(index[k]) for k in keys])
        for key in holdout:
            counts[keys.index(key)] = 0
        weights = counts / counts.sum()

        self.index = index
        self.keys = keys
        self.counts = defaultdict(
            lambda: 0, 
            {key: counts[keys.index(key)] for key in keys}
        )
        self.weights = weights
        if len(holdout) == 0:
            holdout = keys
        self.holdout = holdout

        self.sep = sep

    def realize(self, seq, names):
        dec = list(self.vocab.decode(seq))
        out = []
        for w in dec:
            if w in names:
                out += list(names[w])
            else:
                out.append(w)
        return out

    def _sample(self, key=None, wug_limit=None):
        if key is None:
            i_key = np.random.choice(len(self.keys), p=self.weights)
            key = self.keys[i_key]
        group = list(self.index[key])
        c1, c2 = group[np.random.randint(len(group))]
        o1, o2 = group[np.random.randint(len(group))]

        ctx = (c1 + (sep,) + c2)
        out = (o1 + (sep,) + o2)

        names = {}

        if wug_limit is None:
            wug0 = wug_template % 0
            ctx = t_replace_all(key, (wug0,), ctx)
            out = t_replace_all(key, (wug0,), out)
            names[wug0] = key
            for i_wug in range(1, np.random.randint(max_wugs)+1):
                word = ctx[np.random.randint(len(ctx))]
                if word == sep:
                    continue
                wug = wug_template % i_wug
                ctx = t_replace_all((word,), (wug,), ctx)
                out = t_replace_all((word,), (wug,), out)
                names[wug] = word
        else:
            unk_indices = [
                i for i, w in enumerate(c1) 
                if w != sep and self.counts[(w,)] < wug_limit
            ]
            if len(unk_indices) >= max_wugs:
                unk_indices = unk_indices[:max_wugs]
                print("warning: found a sentence with %d wugs" % len(unk_indices))
            for i_wug, i in enumerate(unk_indices):
                wug = wug_template % i_wug
                word = (ctx[i],)
                ctx = t_replace_all(word, (wug,), ctx)
                out = t_replace_all(word, (wug,), out)
                names[wug] = word
        print(names)

        ctx = self.vocab.encode(ctx)
        out = self.vocab.encode(out)

        return ctx, out, names

    def sample_train(self, wug_limit=None):
        return self._sample(wug_limit=wug_limit)

    def sample_holdout(self, wug_limit=None):
        keys = list(self.holdout)
        key = keys[np.random.randint(len(keys))]
        return self._sample(key=key, wug_limit=wug_limit)
