from torchdec.vocab import Vocab

import numpy as np

MAX_LEN = 20

class RgFactory(object):
    max_states = 10
    max_symbols = 10
    edge_prob = 0.3
    accept_prob = 0.2

    def __call__(self):
        while True:
            samp = self._try_call()
            if samp is not None:
                return samp

    def _try_call(self):
        n_states = self.n_states()
        n_symbols = self.n_symbols()

        states = {}
        for i in range(n_states):
            states[i] = self.is_acceptor()

        edges = {}
        for i in range(n_states):
            for j in range(n_states):
                if not self.is_edge():
                    continue
                sym = self.symbol(n_symbols)
                edges[i, j] = sym

        pstates, pedges = self.prune(states, edges)

        if len(pedges) == 0:
            return None
        return Rg(pstates, pedges)

    def vocab(self):
        vocab = Vocab()
        for i in range(self.max_symbols):
            vocab.add(self.symbol_str(i))
        return vocab

    def n_states(self):
        return np.random.randint(1, self.max_states+1)

    def n_symbols(self):
        return np.random.randint(1, self.max_symbols+1)

    def is_edge(self):
        return bool(np.random.binomial(n=1, p=self.edge_prob))

    def is_acceptor(self):
        return bool(np.random.binomial(n=1, p=self.accept_prob))

    def symbol(self, n_symbols):
        return self.symbol_str(np.random.randint(n_symbols))

    def symbol_str(self, i):
        apart = i % 26
        npart = i // 26
        if self.max_symbols <= 26:
            a = chr(ord('a') + apart)
        else:
            a = chr(ord('a') + apart) + str(npart)
        return a

    def prune(self, states, edges):
        visited = set()
        start_reaches = set()
        queue = [0]
        while len(queue) > 0:
            state = queue.pop()
            if state in visited:
                continue
            visited.add(state)
            start_reaches.add(state)
            children = [i for i in states if (state, i) in edges]
            queue += children

        # TODO eugh
        visited = set()
        reaches_end = set()
        queue = [state for state, accept in states.items() if accept]
        while len(queue) > 0:
            state = queue.pop()
            if state in visited:
                continue
            visited.add(state)
            reaches_end.add(state)
            parents = [i for i in states if (i, state) in edges]
            queue += parents

        reachable = start_reaches & reaches_end

        pruned_states = {
            state: accept for state, accept in states.items()
            if state in reachable
        }
        pruned_edges = {
            (e1, e2): label for (e1, e2), label in edges.items()
            if e1 in reachable and e2 in reachable
        }
        return pruned_states, pruned_edges

class Rg(object):
    def __init__(self, states, edges):
        self.states = states
        self.edges = edges
        self._pretty_states = {
            state: ("%d*" % state if reachable else str(state))
            for state, reachable in states.items()
        }

    def pp(self):
        out = []
        for (i, j), label in self.edges.items():
            out.append("%-2s -- %s --> %-2s" % (
                self._pretty_states[i], label, self._pretty_states[j]
            ))
        return "\n".join(out)

    def __call__(self):
        state = 0
        out = []
        for _ in range(MAX_LEN):
            nexts = [i for i in self.states if (state, i) in self.edges]
            if self.states[state] == True:
                nexts.append(None)
            new_state = np.random.choice(nexts)
            if new_state is None:
                break
            else:
                label = self.edges[state, new_state]
                out.append(label)
                state = new_state
        return out

    def accepts(self, str):
        pstate = {0}
        for tok in str:
            nexts = {
                i for i in self.states 
                if any(
                    (s, i) in self.edges and self.edges[s, i] == tok
                    for s in pstate
                )
            }
            if len(nexts) == 0:
                return False
            pstate = nexts
        return any(self.states[s] == True for s in pstate)

class CfgFactory(object):
    max_nonterminals = 50
    max_terminals = 20
    max_edges = 100

    def vocab(self):
        vocab = Vocab()
        for t in range(self.max_terminals):
            vocab.add(self.word_str(t))
        vocab.add("(")
        vocab.add(")")
        return vocab

    def is_bracket(self, c):
        return c == "(" or c == ")"

    def n_nonterminals(self):
        return np.random.randint(1, self.max_nonterminals+1)

    def n_terminals(self):
        return np.random.randint(1, self.max_terminals+1)

    def n_edges(self):
        return np.random.randint(1, self.max_edges+1)

    def lhs(self, active, n_nt):
        available = [a for a in active if a < n_nt]
        return np.random.choice(available)

    def rhs(self, lhs, n_nt, n_t):
        return np.random.randint(lhs + 1, n_nt + n_t)

    def word(self, t, n_nt, n_t):
        i = t - n_nt
        return self.word_str(i)

    def word_str(self, i):
        apart = i % 26
        npart = i // 26
        a = chr(ord('a') + apart) + str(npart)
        return a

    def prune(self, edges, n_nt):
        cache = {}
        def rec(sym):
            if sym in cache:
                return cache[sym]
            if sym >= n_nt:
                cache[sym] = True
                return True
            sym_edges = {e for e in edges if e[0] == sym}
            if len(sym_edges) == 0:
                cache[sym] = False
                return False
            down = {rec(e[1][0]) and rec(e[1][1]) for e in sym_edges}
            result = any(down)
            cache[sym] = result
            return result
        rec(0)
        def ok(rule):
            l, (r1, r2) = rule
            return all(s in cache and cache[s] for s in (l, r1, r2))

        return {e for e in edges if ok(e)}

    def _try_call(self):
        n_nt = self.n_nonterminals()
        n_t = self.n_terminals()
        n_edges = self.n_edges()
        if (n_nt * (n_nt + n_t)) < n_edges:
            return None
        
        edges = set()
        active = [0]
        while len(edges) < n_edges:
            lhs = self.lhs(active, n_nt)
            rhs1 = self.rhs(lhs, n_nt, n_t)
            rhs2 = self.rhs(lhs, n_nt, n_t)
            edges.add((lhs, (rhs1, rhs2)))
            if rhs1 not in active:
                active.append(rhs1)
            if rhs2 not in active:
                active.append(rhs2)
        edges = self.prune(edges, n_nt)
        if len(edges) == 0:
            return None
        if not any(e[0] == 0 for e in edges):
            return None

        words = set()
        for t in range(n_nt, n_nt + n_t):
            words.add((t, self.word(t, n_nt, n_t)))

        return Grammar(edges, words)

    def __call__(self):
        while True:
            samp = self._try_call()
            if samp is not None:
                return samp

class Cfg(object):
    def __init__(self, edges, words):
        self.nt_symbols = {e[0] for e in edges}
        self.t_symbols = {e[0] for e in words}
        assert len(self.nt_symbols & self.t_symbols) == 0

        self.edges = {
            s: [r for l, r in edges if l == s]
            for s in self.nt_symbols
        }
        self.words = {
            s: [r for l, r in words if l == s]
            for s in self.t_symbols
        }

    def pp(self):
        out = []
        for l, rs in sorted(self.edges.items()):
            out.append("%s ->" % l)
            for r1, r2 in rs:
                out.append("  %s %s" % (r1, r2))
        out.append("")
        for l, ws in sorted(self.words.items()):
            out.append("%s ->" % l)
            for w in ws:
                out.append("  %s" % w)
        return ("\n").join(out)

    def __call__(self, sym=0, bracket=False):
        if sym in self.nt_symbols:
            edges = self.edges[sym]
            s1, s2 = edges[np.random.randint(len(edges))]
            r1 = self(s1, bracket)
            r2 = self(s2, bracket)
            return ("(",) + r1 + r2 + (")",) if bracket else r1 + r2
        elif sym in self.t_symbols:
            word = np.random.choice(self.words[sym])
            return (word,)
        else:
            print(self.edges)
            print(self.words)
            print(sym)
            assert False, "unreachable"
