from collections import defaultdict, namedtuple
import heapq
import numpy as np

IndexResult = namedtuple("IndexResult", "key values score")

class FuzzyIndex(object):
    def __init__(self, ngram_size=3, tfidf=False):
        self.key_part_to_key = defaultdict(set)
        self.key_to_value = defaultdict(set)
        self.doc_freq = defaultdict(lambda: 0)
        self.doc_count = 0

        self.ngram_size = ngram_size
        self.tfidf = tfidf

    def put(self, key, value):
        if value in self.key_to_value[key]:
            return
        self.key_to_value[key].add(value)
        parts = self.make_parts(key)
        for part in parts:
            self.key_part_to_key[part].add(key)
            self.doc_freq[part] += 1
            self.doc_count += 1

    def get(self, key, threshold):
        parts = self.make_parts(key)
        candidates = defaultdict(lambda: 0)
        for part in parts:
            if self.tfidf:
                score = np.log(self.doc_count) - np.log(self.doc_freq[part])
            else:
                score = 1
            for cand in self.key_part_to_key[part]:
                candidates[cand] += score
        ret = []
        for k, count in candidates.items():
            if count >= threshold:
                ret.append(IndexResult(k, self.key_to_value[k], count))
        return ret

    def make_parts(self, key):
        return tuple(
            tuple(key[i:i+self.ngram_size]) 
            for i in range(len(key)-self.ngram_size)
        )
