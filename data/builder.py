from collections import defaultdict
import numpy as np
from torchdec import hlog
from torchdec.vocab import Vocab

#max_wugs = 2
max_size = 2
wug_template = "WUG%d"
wug1 = wug_template % 0
wug2 = wug_template % 1
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

        logcounts = np.log(counts)
        inv_weights = np.max(logcounts) + 1 - logcounts
        inv_weights = inv_weights / inv_weights.sum()
        #for i in inv_weights.argsort():
        #    print(counts[i], inv_weights[i], keys[i])
        #exit()

        self.index = index
        self.keys = keys
        self.counts = defaultdict(
            lambda: 0, 
            {key: counts[keys.index(key)] for key in keys}
        )
        self.weights = weights
        self.inv_weights = inv_weights
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

    def novel(self, inp=None, out=None):
        if inp is None:
            out = tuple(out)
            return not any(o == out for i, o in self.train_utts)
        if out is None:
            inp = tuple(inp)
            return not any(i == inp for i, o in self.train_utts)
        return (tuple(inp), tuple(out)) not in self.train_utts

    def realize(self, seq, names):
        dec = list(self.vocab.decode(seq))
        out = []
        for w in dec:
            if w in names:
                out += list(names[w])
            else:
                out.append(w)
        return out

    def _compute_adjacencies(self):
        for utt in self.train_utts

    def _make_generic(self, seq):
        for span1 in range(1, max_size+1):
            for i in range(len(seq)+1-span1):
                arg1 = seq[i:i+span1]
                templ1 = t_replace_all(arg1, (wug1,), seq)
                if sep in arg1:
                    continue
                yield (
                    tuple(self.vocab.encode(templ1)),
                    (tuple(self.vocab.encode(arg1)[1:-1]),)
                )
                for span2 in range(1, max_size+1):
                    for j in range(i, len(seq)+1-span2):
                        if i+span1 > j:
                            continue
                        arg2 = seq[j:j+span2]
                        if sep in arg2:
                            continue
                        templ2 = t_replace_all(arg2, (wug2,), templ1)
                        yield (
                            tuple(self.vocab.encode(templ2)),
                            (
                                tuple(self.vocab.encode(arg1)[1:-1]),
                                tuple(self.vocab.encode(arg2)[1:-1])
                            )
                        )

    def enumerate_comp_train(self):
        assert max_wugs == 2
        wug1 = wug_template % 0
        wug2 = wug_template % 1
        for utt in self.train_utts:
            inp, out = utt
            seq = inp + (sep,) + out
            for generics in self._make_generic(seq):
                yield generic

            #for span in range(1, max_size+1):
            #    for i in range(len(seq)+1-span):
            #        argument = seq[i:i+span]
            #        template = t_replace_all(argument, (wug_template % 0,), seq)
            #        #template = seq[:i] + (wug_template % 0,) + seq[i+1:]
            #        yield (
            #            tuple(self.vocab.encode(template)),
            #            tuple(self.vocab.encode(argument)[1:-1])
            #        )

    def _sample(self, utts, index=None):
        if index is None:
            index = np.random.randint(len(utts))
        inp, out = utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out

    def _sample_comp(self, key=None, wug_limit=None, inv_weights=False):
        weights = self.inv_weights if inv_weights else self.weights
        #print(inv_weights)
        if key is None:
            i_key = np.random.choice(len(self.keys), p=weights)
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
                i for i, w in enumerate(ctx)
                if w != sep and self.counts[(w,)] < wug_limit 
                and w not in ctx[:i]
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

        #print(ctx, names)

        ctx = self.vocab.encode(ctx)
        out = self.vocab.encode(out)

        return ctx, out, names

    def sample_comp_train(self, wug_limit=None):
        return self._sample_comp(wug_limit=wug_limit)[:2]

    def sample_comp_gen(self, wug_limit=None):
        #print(self.holdout)
        #keys = list(self.holdout)
        #key = keys[np.random.randint(len(keys))]
        return self._sample_comp(wug_limit=wug_limit, inv_weights=True)

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
