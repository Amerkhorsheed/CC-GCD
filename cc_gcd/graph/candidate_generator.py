"""KGE candidate generator: absent edges that the KG-Trie may traverse (plan section 5.1).

A hard KG-Trie (GCR, DoG) is built from observed paths only, so an absence
penalty can never apply unless absent edges are proposed explicitly. For every
entity u this module proposes

    top-R relations r for u  x  top-k tails v of (u, r, ?) under ComplEx,

excluding edges already observed (they are in the trie anyway) and self loops.
Relations live in the inverse-augmented space (r and ~r), the same space the
trie walks, so a candidate (u, ~r, v) stands for the absent fact (v, r, u).

Relation ranking (``rel_rank``):
    "kge"       score(u, r) = best raw KGE score among admissible absent tails of (u, r).
                Raw ComplEx scores are not comparable across relations; kept as an ablation.
    "kge_prob"  score(u, r) = log P(best admissible absent tail | u, r) under the model's
                1-vs-all softmax over all tails (observed tails included in the normalizer).
    "cooc"      score(u, r) = max over relations r' observed at u of
                P(an entity has r | it has r'), from the observed graph
                (relation co-occurrence, in the spirit of Galarraga et al. 2017).
    "hybrid"    log cooc(u, r) + kge_prob score.
Each stored candidate keeps its raw score and its log P(v | u, r) (``tail_logp``), the
uncalibrated input for the plausibility prior pi(e) (calibration/kge_prior.py, next step).

Signature filter (``signature_filter``): a relation r is admissible for u only if
cooc(u, r) > tau, and a tail v is admissible for r only if cooc(v, ~r) > tau,
i.e. some entity with v's observed relations also has ~r. BRINK ships no trusted
ontology, so this is a data-derived *candidate filter*, not the -inf ontology
mask of plan section 4.2 (an unknown type is never treated as a violation there).

Everything is computed from the observed graph and a KGE trained on it, never
from removed triples or test questions. ``build`` runs once per graph on the
GPU and stores the top R_max x k_max table; R and k are sliced at query time.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import AbstractSet, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from cc_gcd.kge.complex_kge import ComplExScorer, augment, inverse

Triple = Tuple[str, str, str]
RELATION_RANKINGS = ("kge", "kge_prob", "hybrid", "cooc")


class Candidate(tuple):
    """(head, relation, tail, kge_score, relation_rank, tail_rank); ranks are 0-based."""
    __slots__ = ()

    def __new__(cls, h, r, t, score, rel_rank, tail_rank):
        return super().__new__(cls, (h, r, t, score, rel_rank, tail_rank))


class CandidateGenerator:
    def __init__(self, triples: Sequence[Triple], scorer: ComplExScorer, *, rel_rank: str = "kge",
                 signature_filter: bool = True, tau: float = 0.0, R_max: int = 10, k_max: int = 20):
        if rel_rank not in RELATION_RANKINGS:
            raise ValueError(f"rel_rank must be one of {RELATION_RANKINGS}")
        self.sc = scorer
        self.rel_rank, self.signature_filter, self.tau = rel_rank, signature_filter, tau
        self.R_max, self.k_max = R_max, k_max
        E, Rr = len(scorer.entities), len(scorer.relations)
        self.E, self.Rr = E, Rr
        self.inv_rel = torch.tensor([scorer.r2id[inverse(r)] for r in scorer.relations])

        # Observed inverse-augmented adjacency, by id.
        self.obs: Dict[Tuple[int, int], np.ndarray] = {}
        tmp: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for h, r, t in set(augment(triples)):
            tmp[(scorer.e2id[h], scorer.r2id[r])].append(scorer.e2id[t])
        self.obs = {k: np.asarray(v, dtype=np.int64) for k, v in tmp.items()}

        # Relation signatures and co-occurrence confidence conf[r', r] = P(has r | has r').
        M = torch.zeros(E, Rr)
        for (u, r) in self.obs:
            M[u, r] = 1.0
        co = M.T @ M
        conf = co / co.diag().clamp(min=1).unsqueeze(1)
        self.cooc = self._max_product(M, conf)  # (E, Rr) on CPU, float32
        self._range_ok = (self.cooc.T > self.tau)  # (Rr, E): v's signature is compatible with relation
        self.table: dict = {}

    # ------------------------------------------------------------------ build
    @staticmethod
    def _max_product(M: torch.Tensor, conf: torch.Tensor, chunk: int = 64) -> torch.Tensor:
        out = torch.empty_like(M)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        conf_d = conf.to(dev)
        for i in range(0, M.shape[0], chunk):
            m = M[i:i + chunk].to(dev)
            out[i:i + chunk] = (m.unsqueeze(2) * conf_d.unsqueeze(0)).amax(1).cpu()
        return out

    def _admissible(self, us: torch.Tensor, rs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """Mask observed tails, self loops and (optionally) signature-incompatible tails with -inf."""
        dev = scores.device
        rows, cols = [], []
        for i, (u, r) in enumerate(zip(us.tolist(), rs.tolist())):
            t = self.obs.get((u, r))
            if t is not None:
                rows.append(np.full(len(t), i))
                cols.append(t)
        if rows:
            scores[torch.from_numpy(np.concatenate(rows)).to(dev), torch.from_numpy(np.concatenate(cols)).to(dev)] = -torch.inf
        scores[torch.arange(len(us), device=dev), us.to(dev)] = -torch.inf
        if self.signature_filter:
            range_ok = self._range_ok[self.inv_rel[rs]].to(dev)  # (B, E): cooc(v, ~r) > tau
            scores.masked_fill_(~range_ok, -torch.inf)
        return scores

    @torch.no_grad()
    def build(self, batch_pairs: int = 2048, progress=None) -> dict:
        E, Rr, Rm, km = self.E, self.Rr, self.R_max, self.k_max
        domain_ok = self.cooc > self.tau if self.signature_filter else torch.ones(E, Rr, dtype=torch.bool)

        rel = np.full((E, Rm), -1, dtype=np.int32)
        rel_score = np.full((E, Rm), -np.inf, dtype=np.float32)
        tail = np.full((E, Rm, km), -1, dtype=np.int32)
        tail_score = np.full((E, Rm, km), -np.inf, dtype=np.float32)
        tail_logp = np.full((E, Rm, km), -np.inf, dtype=np.float32)

        def scores(u, r):
            raw = self.sc.score_tails(u.to(self.sc.device), r.to(self.sc.device))
            lse = torch.logsumexp(raw, dim=1, keepdim=True)  # over *all* tails, observed ones included
            return self._admissible(u, r, raw), lse

        if self.rel_rank == "cooc":
            s = self.cooc.clone()
            s[~domain_ok] = -torch.inf
            top = s.topk(min(Rm, Rr), dim=1)
            pairs_u = torch.arange(E).repeat_interleave(top.indices.shape[1])
            pairs_r = top.indices.reshape(-1)
            rel[:, :top.indices.shape[1]] = top.indices.numpy()
            rel_score[:, :top.indices.shape[1]] = top.values.numpy()
            it = range(0, len(pairs_u), batch_pairs)
            for i in (progress(it) if progress else it):
                u, r = pairs_u[i:i + batch_pairs], pairs_r[i:i + batch_pairs]
                sc, lse = scores(u, r)
                tv = sc.topk(km, dim=1)
                flat = np.arange(i, i + len(u))
                tail.reshape(-1, km)[flat] = tv.indices.cpu().numpy()
                tail_score.reshape(-1, km)[flat] = tv.values.cpu().numpy()
                tail_logp.reshape(-1, km)[flat] = (tv.values - lse).cpu().numpy()
        else:  # KGE-based: score every (u, r), keep the R_max relations with the best admissible tail
            U = max(1, batch_pairs // Rr)
            it = range(0, E, U)
            log_cooc = torch.log(self.cooc.clamp(min=1e-12))
            for i in (progress(it) if progress else it):
                us = torch.arange(i, min(i + U, E))
                u = us.repeat_interleave(Rr)
                r = torch.arange(Rr).repeat(len(us))
                sc, lse = scores(u, r)
                tv = sc.topk(km, dim=1)
                vals = tv.values.view(len(us), Rr, km)
                logp = (tv.values - lse).view(len(us), Rr, km)
                idx = tv.indices.view(len(us), Rr, km)
                if self.rel_rank == "kge":  # raw score of the best absent tail
                    key = vals[:, :, 0]
                elif self.rel_rank == "kge_prob":  # log P(best absent tail | u, r) under the 1-vs-all softmax
                    key = logp[:, :, 0]
                else:  # "hybrid": log cooc(u, r) + log P(best absent tail | u, r)
                    key = logp[:, :, 0] + log_cooc[us].to(logp.device)
                best = key.masked_fill(~domain_ok[us].to(vals.device), -torch.inf)
                rt = best.topk(min(Rm, Rr), dim=1)
                g = rt.indices.unsqueeze(2).expand(-1, -1, km)
                rel[i:i + len(us), :rt.indices.shape[1]] = rt.indices.cpu().numpy()
                rel_score[i:i + len(us), :rt.indices.shape[1]] = rt.values.cpu().numpy()
                tail[i:i + len(us), :rt.indices.shape[1]] = idx.gather(1, g).cpu().numpy()
                tail_score[i:i + len(us), :rt.indices.shape[1]] = vals.gather(1, g).cpu().numpy()
                tail_logp[i:i + len(us), :rt.indices.shape[1]] = logp.gather(1, g).cpu().numpy()

        # Domain-filtered relations and fully masked tails contribute nothing.
        rel[~np.isfinite(rel_score)] = -1
        tail[rel < 0] = -1
        tail[~np.isfinite(tail_score)] = -1
        self.table = {"rel": rel, "rel_score": rel_score, "tail": tail, "tail_score": tail_score,
                      "tail_logp": tail_logp}
        return self.table

    # ------------------------------------------------------------------ io
    def config(self) -> dict:
        return {"rel_rank": self.rel_rank, "signature_filter": self.signature_filter, "tau": self.tau,
                "R_max": self.R_max, "k_max": self.k_max}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.table)
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump(self.config(), f, indent=2)

    def load(self, path: str) -> "CandidateGenerator":
        z = np.load(path)
        self.table = {k: z[k] for k in z.files}
        return self

    # ------------------------------------------------------------------ query
    def candidates(self, u: str, R: int, k: int) -> List[Candidate]:
        """Absent edges (u, r, v) for the top-R relations and top-k tails, canonical ids."""
        ui = self.sc.e2id.get(u)
        if ui is None:
            return []
        T = self.table
        out = []
        for a in range(min(R, self.R_max)):
            ri = int(T["rel"][ui, a])
            if ri < 0:
                continue
            r = self.sc.relations[ri]
            for b in range(min(k, self.k_max)):
                vi = int(T["tail"][ui, a, b])
                if vi < 0:
                    break
                out.append(Candidate(u, r, self.sc.entities[vi], float(T["tail_score"][ui, a, b]), a, b))
        return out

    def expand(self, seeds: Iterable[str], adj: Dict[str, List[Tuple[str, str]]], R: int, k: int,
               cand_hops: AbstractSet[int] = frozenset({1}), hops: int = 2) -> List[Candidate]:
        """Candidate edges for one question: at hop d in ``cand_hops`` every node reached at depth d-1
        (through observed or candidate edges) proposes its top-R x k absent edges."""
        frontier, seen, out = set(seeds), set(seeds), []
        for d in range(1, hops + 1):
            nxt = set()
            for u in frontier:
                nxt.update(v for _, v in adj.get(u, []))
                if d in cand_hops:
                    c = self.candidates(u, R, k)
                    out.extend(c)
                    nxt.update(x[2] for x in c)
            frontier = nxt - seen
            seen |= nxt
        return out

    # ------------------------------------------------------------------ diagnostics
    @torch.no_grad()
    def tail_rank(self, u: str, r: str, v: str) -> int:
        """Filtered 0-based rank of v among admissible absent tails of (u, r); -1 if v is masked."""
        ui, ri, vi = self.sc.e2id[u], self.sc.r2id[r], self.sc.e2id[v]
        us, rs = torch.tensor([ui]), torch.tensor([ri])
        s = self._admissible(us, rs, self.sc.score_tails(us.to(self.sc.device), rs.to(self.sc.device)))[0]
        if not torch.isfinite(s[vi]):
            return -1
        return int((s > s[vi]).sum().item())

    def relation_rank(self, u: str, r: str) -> int:
        """0-based rank of r among u's stored relations; -1 if r is outside the top R_max."""
        ui, ri = self.sc.e2id[u], self.sc.r2id[r]
        hit = np.where(self.table["rel"][ui] == ri)[0]
        return int(hit[0]) if len(hit) else -1
