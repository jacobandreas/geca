from collections import Counter, defaultdict
import numpy as np
from torchdec import hlog
from torchdec.vocab import Vocab
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_boolean("compute_adjacencies", False, "compute adjacencies")
flags.DEFINE_boolean("compute_alignments", False, "")
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
        if FLAGS.compute_alignments:
            self._compute_alignments(train_utts)
        if FLAGS.compute_adjacencies:
            self._compute_adjacencies(train_utts)

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

    def _compute_alignments(self, utts):
        numerator = Counter()
        denominator = Counter()
        count = 0
        for inp, out in utts:
            count += 1
            for o in set(out):
                denominator[o] += 1
            for i in set(inp):
                denominator[i] += 1
                for o in set(out):
                    numerator[i, o] += 1
        for i, o in numerator:
            pmi = np.log(numerator[i, o]) - np.log(denominator[i]) - np.log(denominator[o])
            pmi -= np.log(sum(numerator.values())) - 2 * np.log(sum(denominator.values()))
            #both = numerator[i, o] / count
            #one = (denominator[i] + denominator[o] - numerator[i, o]) / count
            #neither = 1 - both - neither

            #cov =  (numerator[i,o] / count) - (denominator[i] / count * denominator[o] / count)
            score = pmi
            if score > -10000:
                print(i, o, score)
        exit()

    def _compute_adjacencies(self, utts):
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
                #seq = inp
                for generic in self._make_generic(seq):
                    yield generic
        counts = Counter()
        templ_to_arg = defaultdict(set)
        arg_to_templ = defaultdict(set)
        for templ, args in enumerate():
            if any(a not in keep_args for a in args):
                continue
            templ_to_arg[templ].add(args)
            arg_to_templ[args].add(templ)

        self.templates = []
        for templ, args in templ_to_arg.items():
            assert all(a in keep_args for aa in args for a in aa)
            if any(len(arg_to_templ[a]) > 1 for a in args):
                self.templates.append(templ)
        self.templates = sorted(self.templates)

        self.arg_to_templ = {k: sorted(list(v)) for k, v in arg_to_templ.items()}
        self.templ_to_arg = {k: sorted(list(v)) for k, v in templ_to_arg.items()}
        #self.args = sorted(list([k for k in self.arg_to_templ.keys()]))
        #weights = np.asarray([len(templ_to_arg[t]) for t in self.templates])
        weights = np.zeros(len(self.templates))
        for arg, templs in self.arg_to_templ.items():
            for templ in templs:
                if templ in self.templates:
                    weights[self.templates.index(templ)] += 1
        weights = weights / weights.sum()
        self.weights = weights

        hlog.log("LOADED")

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
                    for j in range(i+1, len(seq)+1-span2):
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

    def split(self, templ):
        return t_split(self.vocab[self.sep], templ, self.vocab)

    def join(self, templ):
        inp, out = templ
        return inp[:-1] + (self.vocab[self.sep],) + out[1:]

    def enumerate_comp(self):
        for arg1 in self.arg_to_templ:
            generalizations = []
            for templ1 in self.arg_to_templ[arg1]:
                for arg2 in self.templ_to_arg[templ1]:
                    for templ2 in self.arg_to_templ[arg2]:
                        if arg1 not in self.templ_to_arg[templ2]:
                            generalizations.append((templ1, templ2))

            if len(generalizations) > 0:
                dec_arg = [self.vocab.decode(a) for a in arg1]
                names = dict(zip([wug1, wug2], dec_arg))
                np.random.shuffle(generalizations)
                for templ1, templ2 in generalizations:
                    yield self.split(templ1), self.split(templ2), names
                    #yield templ1, templ2, names
                    break

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
