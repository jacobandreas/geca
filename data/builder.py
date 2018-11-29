from fuzzy_index import FuzzyIndex

from collections import Counter, defaultdict
import numpy as np
from torchdec import hlog
from torchdec.vocab import Vocab
from absl import flags
import heapq

FLAGS = flags.FLAGS
flags.DEFINE_boolean("compute_adjacencies", False, "compute adjacencies")
flags.DEFINE_boolean("dedup", False, "deduplicate training examples")
flags.DEFINE_integer("wug_limit", None, "wug limit")

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

def t_split(sep, seq, vocab):
    index = seq.index(sep)
    inp, out = seq[:index], seq[index+1:]
    assert inp[0] == vocab.sos()
    assert vocab.eos() not in inp
    inp = inp + (vocab.eos(),)
    assert out[-1] == vocab.eos()
    assert vocab.sos() not in out
    out = (vocab.sos(),) + out
    return inp, out

class OneShotDataset(object):
    def __init__(
            self,
            train_utts,
            val_utts,
            test_utts,
            aug_data=(),
            invert=False,
    ):
        vocab = Vocab()
        vocab.add(wug1)
        vocab.add(wug2)
        vocab.add(sep)
        for utts in (train_utts, val_utts, test_utts):
            for inp, out in utts:
                for seq in (inp, out):
                    for word in seq:
                        vocab.add(word)

        aug_utts = [(tuple(d["inp"]), tuple(d["out"])) for d in aug_data]
        if FLAGS.dedup:
            train_utts = [(tuple(i), tuple(o)) for i, o in train_utts]
            train_utts = sorted(list(set(train_utts)))
        hlog.value("train", len(train_utts))
        hlog.value("aug", len(aug_utts))

        if invert:
            train_utts = [(o, i) for i, o in train_utts]
            aug_utts = [(o, i) for i, o in aug_utts]
            val_utts = [(o, i) for i, o in val_utts]
            test_utts = [(o, i) for i, o in test_utts]

        self.vocab = vocab
        self.sep = sep
        self.train_utts = train_utts
        self.aug_utts = aug_utts
        self.val_utts = val_utts
        self.test_utts = test_utts
        if FLAGS.compute_adjacencies:
            self._compute_similarities(train_utts)

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
        return tuple(out)

    def _compute_similarities(self, utts):
        counts = Counter()
        for utt in utts:
            inp, out = utt
            for seq in (inp, out):
                enc = self.vocab.encode(seq)[1:-1]
                for span in range(1, max_size+1):
                    for i in range(len(enc)+1-span):
                        counts[tuple(enc[i:i+span])] += 1
        keep_args = set([c for c, n in counts.items() if n <= FLAGS.wug_limit])

        def enumerate():
            for utt in utts:
                inp, out = utt
                seq = inp + (sep,) + out
                #yield tuple(self.vocab.encode(seq))
                for generic in self._make_generic(seq, keep_args):
                #    yield generic + (tuple(self.vocab.encode(seq)),)
                    yield generic, utt

        arg_to_templ = defaultdict(set)
        templ_to_arg = defaultdict(set)
        templ_to_templ = defaultdict(set)
        sim_templ = FuzzyIndex(tfidf=True)
        templ_to_orig = defaultdict(set)
        for (templ, args), orig in enumerate():
            arg_to_templ[args].add(templ)
            templ_to_arg[templ].add(args)
            sim_templ.put(templ, args)
            templ_to_orig[templ].add(orig)

        multiplicity = defaultdict(lambda: 0)
        for args1 in arg_to_templ:
            for templ1 in arg_to_templ[args1]:
                multiplicity[templ1] += 1
                #alt_args = [
                #    a for a in templ_to_arg[templ1] 
                #    if len(set(a) & set(args1)) == 0
                #]
                #if len(alt_args) == 0:
                #    continue
                for templ2 in arg_to_templ[args1]:
                    if templ1 == templ2:
                        continue
                    if (templ1, templ2) in templ_to_templ:
                        continue
                    #if all(args2 in templ_to_arg[templ2] for args2 in alt_args):
                    #    continue
                    templ_to_templ[templ2].add(templ1)

        #for templ, count in sorted(multiplicity.items(), key=lambda p: -p[1]):
        #    #if count > 1:
        #    #    continue
        #    n_args = len(next(iter(templ_to_arg[templ])))
        #    for sim in sorted(sim_templ.get(templ, 0), key=lambda s: -s.score):
        #        if sim.score < 4.5:
        #            continue
        #        if len(next(iter(sim.values))) != n_args:
        #            continue
        #        if templ_to_orig[templ] == templ_to_orig[sim.key]:
        #            continue
        #        print(count, " ".join(self.vocab.decode(templ)))
        #        print(sim.score, "~ " + " ".join(self.vocab.decode(sim.key)))
        #        print()
        #        break

        self.templ_to_arg = templ_to_arg
        self.arg_to_templ = arg_to_templ
        self.templ_to_templ = templ_to_templ
        self.multiplicity = multiplicity

        comp_pairs = []
        for templ1 in self.templ_to_templ:
            if self.multiplicity[templ1] <= 1:
                continue
            for templ2 in self.templ_to_templ[templ1]:
                comp_pairs.append((templ1, templ2))
        self.comp_pairs = comp_pairs

    #def _compute_adjacencies(self, utts):
    #    counts = Counter()
    #    for utt in utts:
    #        inp, out = utt
    #        for seq in (inp, out):
    #            enc = self.vocab.encode(seq)[1:-1]
    #            for span in range(1, max_size+1):
    #                for i in range(len(enc)+1-span):
    #                    counts[tuple(enc[i:i+span])] += 1
    #    keep_args = set([c for c, n in counts.items() if n <= FLAGS.wug_limit])

    #    def enumerate():
    #        for utt in utts:
    #            inp, out = utt
    #            seq = inp + (sep,) + out
    #            #seq = inp
    #            for generic in self._make_generic(seq):
    #                yield generic
    #    counts = Counter()
    #    templ_to_arg = defaultdict(set)
    #    arg_to_templ = defaultdict(set)
    #    for templ, args in enumerate():
    #        if any(a not in keep_args for a in args):
    #            continue
    #        templ_to_arg[templ].add(args)
    #        arg_to_templ[args].add(templ)

    #    self.templates = []
    #    for templ, args in templ_to_arg.items():
    #        assert all(a in keep_args for aa in args for a in aa)
    #        if any(len(arg_to_templ[a]) > 1 for a in args):
    #            self.templates.append(templ)
    #    self.templates = sorted(self.templates)

    #    self.arg_to_templ = {k: sorted(list(v)) for k, v in arg_to_templ.items()}
    #    self.templ_to_arg = {k: sorted(list(v)) for k, v in templ_to_arg.items()}
    #    weights = np.zeros(len(self.templates))
    #    for arg, templs in self.arg_to_templ.items():
    #        for templ in templs:
    #            if templ in self.templates:
    #                weights[self.templates.index(templ)] += 1
    #    weights = weights / weights.sum()
    #    self.weights = weights

    #    hlog.log("LOADED")

    def _make_generic(self, seq, keep):
        for span1 in range(1, max_size+1):
            for i in range(len(seq)+1-span1):
                arg1 = seq[i:i+span1]
                templ1 = t_replace_all(arg1, (wug1,), seq)
                if sep in arg1:
                    continue
                arg1_enc = tuple(self.vocab.encode(arg1)[1:-1])
                if arg1_enc not in keep:
                    continue
                yield (
                    tuple(self.vocab.encode(templ1)),
                    (arg1_enc,)
                )
                for span2 in range(1, max_size+1):
                    for j in range(i+1, len(seq)+1-span2):
                        if i+span1 > j:
                            continue
                        arg2 = seq[j:j+span2]
                        if sep in arg2:
                            continue
                        arg2_enc = tuple(self.vocab.encode(arg2)[1:-1])
                        if arg2_enc not in keep:
                            continue
                        if len(set(arg1_enc) & set(arg2_enc)) > 0:
                            continue

                        templ2 = t_replace_all(arg2, (wug2,), templ1)
                        yield (
                            tuple(self.vocab.encode(templ2)),
                            (arg1_enc, arg2_enc)
                        )

    def split(self, templ):
        return t_split(self.vocab[self.sep], templ, self.vocab)

    def join(self, templ):
        inp, out = templ
        return inp[:-1] + (self.vocab[self.sep],) + out[1:]

    def enumerate_comp(self):
        count = 0
        for templ2 in self.templ_to_templ:
            args = [
                arg 
                for templ1 in self.templ_to_templ[templ2]
                if self.multiplicity[templ1] > 1
                for arg in self.templ_to_arg[templ1]
                if arg not in self.templ_to_arg[templ2]
            ]
            np.random.shuffle(args)
            for arg in args[:2]:
                dec_arg = [self.vocab.decode(a) for a in arg]
                names = dict(zip([wug1, wug2], dec_arg))
                yield self.split(templ2), self.split(templ2), names
                count += 1
        print("%d proposals" % count)

    def _sample_comp(self):
        i = np.random.randint(len(self.comp_pairs))
        return self.comp_pairs[i]

    def sample_comp_train(self):
        templ1, templ2 = self._sample_comp()
        return self.split(templ1), self.split(templ2)

    #def _sample_comp(self):
    #    #weight_arg = self.args[np.random.randint(len(self.args))]
    #    templ1 = self.templates[np.random.choice(len(self.templates), p=self.weights)]
    #    #templs = self.arg_to_templ[weight_arg]
    #    #templ1 = templs[np.random.randint(len(templs))]
    #    templ2 = None
    #    while templ2 is None:
    #        args = self.templ_to_arg[templ1]
    #        arg = args[np.random.randint(len(args))]
    #        neighbors = self.arg_to_templ[arg]
    #        templ2 = neighbors[np.random.randint(len(neighbors))]
    #        if templ2 == templ1:
    #            templ2 = None
    #    names = dict(zip([wug1, wug2], [self.vocab.decode(a) for a in arg]))
    #    templ1 = self.split(templ1)
    #    templ2 = self.split(templ2)
    #    return templ1, templ2, names

    #def sample_comp_train(self):
    #    return self._sample_comp()[:2]

    #def sample_comp_gen(self):
    #    return self._sample_comp()

    def _sample(self, utts, index=None):
        if index is None:
            index = np.random.randint(len(utts))
        inp, out = utts[index]
        inp = self.vocab.encode(inp)
        out = self.vocab.encode(out)
        return inp, out

    def sample_train(self, aug_ratio=0.):
        if np.random.random() < aug_ratio:
            return self._sample(self.aug_utts)
        else:
            return self._sample(self.train_utts)

    def get_val(self):
        return [
            self._sample(self.val_utts, i) for i in range(len(self.val_utts))
        ]

    def get_test(self):
        return [
            self._sample(self.test_utts, i) for i in range(len(self.test_utts))
        ]
