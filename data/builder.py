from collections import defaultdict
import numpy as np
from torchdec import hlog
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
    def __init__(
            self,
            train_utts,
            val_utts,
            test_utts,
            ngram_size=1,
            holdout=frozenset(),
            aug_data=(),
            dedup=False
    ):
        index = defaultdict(set)
        vocab = Vocab()
        for i_wug in range(max_wugs):
            vocab.add(wug_template % i_wug)
        vocab.add(sep)
        for inp, out in train_utts:
            for seq in (inp, out):
                for i in range(len(seq)-ngram_size+1):
                    ngram = tuple(seq[i:i+ngram_size])
                    index[ngram].add((tuple(inp), tuple(out)))
                for word in seq:
                    vocab.add(word)
        for inp, out in val_utts + test_utts:
            for seq in (inp, out):
                for word in seq:
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

        train_utts = train_utts + [(d["inp"], d["out"]) for d in aug_data]
        if dedup:
            train_utts = [(tuple(i), tuple(o)) for i, o in train_utts]
            train_utts = sorted(list(set(train_utts)))
        hlog.value("aug", len(aug_data))
        hlog.value("train+aug", len(train_utts))

        self.sep = sep
        self.train_utts = train_utts
        self.val_utts = val_utts
        self.test_utts = test_utts

    def realize(self, seq, names):
        dec = list(self.vocab.decode(seq))
        out = []
        for w in dec:
            if w in names:
                out += list(names[w])
            else:
                out.append(w)
        return out

    def _sample(self, utts, index=None):
        if index is None:
            index = np.random.randint(len(utts))
        inp, out = utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out

    def _sample_comp(self, key=None, wug_limit=None):
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
                #i for i, w in enumerate(c1) 
                i for i, w in enumerate(ctx)
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

        ctx = self.vocab.encode(ctx)
        out = self.vocab.encode(out)

        return ctx, out, names

    def sample_comp_train(self, wug_limit=None):
        return self._sample_comp(wug_limit=wug_limit)[:2]

    def sample_comp_gen(self, wug_limit=None):
        keys = list(self.holdout)
        key = keys[np.random.randint(len(keys))]
        return self._sample_comp(key=key, wug_limit=wug_limit)

    def sample_train(self):
        return self._sample(self.train_utts)

    def get_val(self):
        return [
            self._sample(self.val_utts, i) for i in range(len(self.val_utts))
        ]

    def get_test(self):
        return [
            self._sample(self.test_utts, i) for i in range(len(self.test_utts))
        ]
