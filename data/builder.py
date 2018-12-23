from fuzzy_index import FuzzyIndex

from absl import flags
from collections import Counter, defaultdict
import heapq
import itertools as it
import numpy as np
from torchdec import hlog
from torchdec.vocab import Vocab

FLAGS = flags.FLAGS
flags.DEFINE_boolean("dedup", False, "deduplicate training examples")
flags.DEFINE_integer("wug_limit", None, "wug limit")
flags.DEFINE_integer("wug_size", 4, "wug size")
flags.DEFINE_integer("wug_count", 2, "number of wugs to insert")
flags.DEFINE_boolean("compute_adjacency", False, "do compositionality")

variants = 5
wug_template = "WUG%d"
def _wugs():
    return [wug_template % i for i in range(FLAGS.wug_count)]

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
        for i in range(FLAGS.wug_count):
            vocab.add(wug_template % i)
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
        if FLAGS.compute_adjacency:
            self._compute_adjacency(train_utts)

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
        used = set()
        for w in dec:
            if w in names:
                used.add(w)
                out += list(names[w])
            else:
                out.append(w)
        return tuple(out), used

    @profile
    def _compute_adjacency(self, utts):
        counts = Counter()
        for utt in utts:
            inp, out = utt
            for seq in (inp, out):
                enc = self.vocab.encode(seq)[1:-1]
                for span in range(1, FLAGS.wug_size+1):
                    for i in range(len(enc)+1-span):
                        counts[tuple(enc[i:i+span])] += 1
        keep_args = set([c for c, n in counts.items() if n <= FLAGS.wug_limit])

        print(len(utts))
        def enumerate_templates():
            for i, utt in enumerate(utts):
                inp, out = utt
                seq = inp + (sep,) + out
                print(i, len(seq))
                for generic in self._make_generic(seq, keep_args):
                    #print(" ".join(self.vocab.decode(generic[0])))
                    yield generic, utt

        arg_to_templ = defaultdict(set)
        templ_to_arg = defaultdict(set)
        templ_to_templ = defaultdict(set)
        #sim_templ = FuzzyIndex(tfidf=True)
        templ_to_orig = defaultdict(set)
        for (templ, args), orig in enumerate_templates():
            arg_to_templ[args].add(templ)
            templ_to_arg[templ].add(args)
            #sim_templ.put(templ, args)
            templ_to_orig[templ].add(orig)

        multiplicity = defaultdict(lambda: 0)
        for args1 in arg_to_templ:
            for templ1 in arg_to_templ[args1]:
                multiplicity[templ1] += 1
                for templ2 in arg_to_templ[args1]:
                    if templ1 == templ2:
                        continue
                    if (templ1, templ2) in templ_to_templ:
                        continue
                    templ_to_templ[templ2].add(templ1)

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
        self.comp_pairs = sorted(comp_pairs)
        self.templates = sorted(self.templ_to_arg.keys())

    def compute_similarity(self, sim_model):
        wugs = [self.vocab[w] for w in _wugs()]
        for templ in self.templates:
            idx = [templ.index(w) for w in wugs if w in templ]

    def _make_generic(self, seq, keep):
        enc_seq = tuple(self.vocab.encode(seq))
        wugs = [self.vocab[w] for w in _wugs()]
        out = self._make_generic_helper(enc_seq, keep, 0, 0, (), wugs)
        return out

    @profile
    def _make_generic_helper(self, seq, keep, begin, i_wug, used_args, wugs):
        for span in range(1, FLAGS.wug_size+1):
            for i in range(begin, len(seq)+1-span):
                arg = seq[i:i+span]
                arg_enc = arg
                templ = t_replace_all(arg, (wugs[i_wug],), seq)
                #templ = seq[:i] + (wugs[i_wug],) + seq[i+span:]
                templ_enc = templ
                if self.vocab[sep] in arg:
                    continue
                #arg_enc = tuple(self.vocab.encode(arg)[1:-1])
                if arg_enc not in keep:
                    continue
                if any(len(set(uarg) & set(arg_enc)) > 0 for uarg in used_args):
                    continue
                next_args = used_args + (arg_enc,)
                assert self.vocab[sep] in templ_enc
                yield (
                    #tuple(self.vocab.encode(templ)),
                    templ_enc,
                    next_args,
                )
                if i_wug+1 < FLAGS.wug_count:
                    for rest in self._make_generic_helper(
                        templ, keep, i+1, i_wug+1, next_args, wugs
                    ):
                        yield rest

    def split(self, templ):
        return t_split(self.vocab[self.sep], templ, self.vocab)

    def join(self, templ):
        inp, out = templ
        return inp[:-1] + (self.vocab[self.sep],) + out[1:]

    def overlap(self, arg, ref_args):
        if arg in ref_args:
            return True
        if all(len(set(arg) & set(a)) > 0 for a in ref_args):
            return True
        return False

    def enumerate_comp(self):
        for templ2 in self.templ_to_templ:
            args2 = self.templ_to_arg[templ2]
            args = [
                arg 
                for templ1 in self.templ_to_templ[templ2]
                if self.multiplicity[templ1] > 1
                for arg in self.templ_to_arg[templ1]
                if not self.overlap(arg, args2)
            ]
            if len(args) == 0:
                continue
            np.random.shuffle(args)
            #args = it.islice(it.chain.from_iterable(it.repeat(args)), variants)
            args = args[:2]
            for arg in args:
                dec_arg = [self.vocab.decode(a) for a in arg]
                names = dict(zip(_wugs(), dec_arg))
                yield self.split(templ2), names

    def enumerate_freq(self):
        for templ, count in self.multiplicity.items():
            if count <= 1:
                continue
            args = list(self.templ_to_arg[templ])
            np.random.shuffle(args)
            for arg in args[:2]:
                dec_arg = [self.vocab.decode(a) for a in arg]
                names = dict(zip(_wugs(), dec_arg))
                yield self.split(templ), names

    def _sample_comp(self):
        i = np.random.randint(len(self.comp_pairs))
        return self.comp_pairs[i]

    def sample_comp_train(self):
        templ1, templ2 = self._sample_comp()
        return self.split(templ1), self.split(templ2)

    def sample_ctx_train(self):
        templ = self.templates[np.random.randint(len(self.templates))]
        args = sorted(self.templ_to_arg[templ])
        arg = args[np.random.randint(len(args))]
        wugs = [self.vocab[w] for w in _wugs()]
        named_arg = list(zip(arg, wugs))
        arg_part, name = named_arg[np.random.randint(len(named_arg))]

        del_arg_part = (self.vocab.sos(),) + arg_part + (self.vocab.eos(),)
        return templ, del_arg_part, templ.index(name)

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

    def get_train(self):
        return [
            self._sample(self.train_utts, i) for i in range(len(self.train_utts))
        ]

    def get_val(self):
        return [
            self._sample(self.val_utts, i) for i in range(len(self.val_utts))
        ]

    def get_test(self):
        return [
            self._sample(self.test_utts, i) for i in range(len(self.test_utts))
        ]
