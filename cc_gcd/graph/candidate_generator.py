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
                softmax over all tails (observed tails included in the normalizer).
    "cooc"      score(u, r) = max over relations r' observed at u of
                P(an entity has r | it has r'), from the observed graph
                (relation co-occurrence, in the spirit of Galarraga et al. 2017).
    "hybrid"    log cooc(u, r) + kge_prob score.
Each stored candidate keeps its raw score and its log P(v | u, r) (``tail_logp``), the
uncalibrated input for the plausibility prior pi(e) (calibration/kge_prior.py, next step).

Signature filter (``signature_filter``): a relation r is admissible for u only if
cooc(u, r) > tau, and a tail v is admissible for r only if cooc(v, ~r) > tau,
i.e. some entity with v's observed relations also has ~r. BRINK and the RoG
subgraphs ship no trusted ontology, so this is a data-derived *candidate filter*,
not the -inf ontology mask of plan section 4.2 (an unknown type is never a violation there).

Scale. Signatures are sparse (entity x relation), co-occurrence is a dense
relation x relation matrix, and the cooc rows and range masks are computed on
demand, so the module runs on BRINK (15k entities, 474 relations) and on the
IKGQA-WebQSP union graph (599k entities, 10k relations) alike. ``build`` can be
restricted to the entities that need candidates (``entities``) and, for the KGE
rankings, to the ``rel_pool`` relations with the highest cooc per entity. With
both left at None the result equals scoring every entity and relation.

Everything is computed from the observed graph and a KGE trained on it, never
from removed triples or test questions. ``build`` runs on the GPU and stores the
top R_max x k_max table; R and k are sliced at query time.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict, defaultdict
from typing import AbstractSet, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from cc_gcd.kge.complex_kge import ComplExScorer, augment, inverse

Triple = Tuple[str, str, str]
RELATION_RANKINGS = ("kge", "kge_prob", "hybrid", "cooc")


class Candidate(tuple):
    """(head, relation, tail, kge_score, relation_rank, tail_rank, log_p); ranks are 0-based,
    log_p = log P(tail | head, relation) under the KGE softmax (input to the prior pi)."""
    __slots__ = ()

    def __new__(cls, h, r, t, score, rel_rank, tail_rank, log_p=float("nan")):
        return super().__new__(cls, (h, r, t, score, rel_rank, tail_rank, log_p))


class CandidateGenerator:
    def __init__(self, triples: Sequence[Triple], scorer: ComplExScorer, *, rel_rank: str = "kge",
                 signature_filter: bool = True, tau: float = 0.0, R_max: int = 10, k_max: int = 20,
                 rel_pool: Optional[int] = None):
        if rel_rank not in RELATION_RANKINGS:
            raise ValueError(f"rel_rank must be one of {RELATION_RANKINGS}")
        self.sc = scorer
        self.dev = scorer.device
        self.rel_rank, self.signature_filter, self.tau = rel_rank, signature_filter, tau
        self.R_max, self.k_max, self.rel_pool = R_max, k_max, rel_pool
        E, Rr = len(scorer.entities), len(scorer.relations)
        self.E, self.Rr = E, Rr
        self.inv_rel = torch.tensor([scorer.r2id[inverse(r)] for r in scorer.relations])

        # Observed inverse-augmented adjacency, by id.
        tmp: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for h, r, t in set(augment(triples)):
            tmp[(scorer.e2id[h], scorer.r2id[r])].append(scorer.e2id[t])
        self.obs: Dict[Tuple[int, int], np.ndarray] = {k: np.asarray(v, dtype=np.int64) for k, v in tmp.items()}

        # Sparse relation signatures M (E x Rr) and co-occurrence confidence conf[r', r] = P(has r | has r').
        keys = np.array(list(self.obs.keys()), dtype=np.int64).reshape(-1, 2)
        self.M = sp.csr_matrix((np.ones(len(keys), dtype=np.float32), (keys[:, 0], keys[:, 1])), shape=(E, Rr))
        co = (self.M.T @ self.M).toarray()
        conf = co / np.maximum(np.diag(co), 1)[:, None]
        self.conf = torch.from_numpy(conf.astype(np.float32)).to(self.dev)
        # Tie-break order for cooc: every relation u already has scores cooc = 1, so ties are common
        # (10th-best == best for 20% of Family and 40% of FB15k-237 entities). Ties go to the relation
        # held by more entities, deterministically and without the KGE.
        freq = np.asarray(self.M.sum(0)).ravel()
        self._rel_order = torch.from_numpy(np.argsort(-freq, kind="stable")).to(self.dev)
        self._range_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        # (E,) bool range masks kept on the GPU, up to ~500 MB: every relation on BRINK, ~800 on IKGQA.
        self._range_cache_size = max(256, int(5e8 // E))
        self.table: dict = {}
        self.row: Dict[int, int] = {}

    # ------------------------------------------------------------------ signatures
    def cooc_rows(self, us: Sequence[int]) -> torch.Tensor:
        """(len(us), Rr) cooc scores: max over u's observed relations r' of conf[r', r]."""
        out = torch.zeros(len(us), self.Rr, device=self.dev)
        for i, u in enumerate(us):
            rels = self.M.indices[self.M.indptr[u]:self.M.indptr[u + 1]]
            if len(rels):
                out[i] = self.conf[torch.from_numpy(rels).to(self.dev)].amax(0)
        return out

    def top_relations(self, cooc: torch.Tensor, domain: torch.Tensor, n: int):
        """Top-n relations per row by cooc (domain-masked), ties broken by relation frequency."""
        key = cooc.masked_fill(~domain, -torch.inf)[:, self._rel_order]
        vals, idx = torch.sort(key, dim=1, descending=True, stable=True)
        return vals[:, :n], self._rel_order[idx[:, :n]]

    def range_ok(self, r: int) -> torch.Tensor:
        """(E,) bool: v's signature is compatible with tails of r, i.e. cooc(v, ~r) > tau."""
        hit = self._range_cache.get(r)
        if hit is None:
            allowed = (self.conf[:, self.inv_rel[r]] > self.tau).cpu().numpy()
            hit = torch.from_numpy(np.asarray(self.M[:, allowed].sum(1)).ravel() > 0).to(self.dev)
            self._range_cache[r] = hit
            if len(self._range_cache) > self._range_cache_size:
                self._range_cache.popitem(last=False)
        else:
            self._range_cache.move_to_end(r)
        return hit

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
            mask = torch.stack([self.range_ok(r) for r in rs.tolist()])
            scores.masked_fill_(~mask, -torch.inf)
        return scores

    def _scores(self, u: torch.Tensor, r: torch.Tensor):
        raw = self.sc.score_tails(u.to(self.dev), r.to(self.dev))
        lse = torch.logsumexp(raw, dim=1, keepdim=True)  # over *all* tails, observed ones included
        return self._admissible(u, r, raw), lse

    # ------------------------------------------------------------------ build
    @torch.no_grad()
    def build(self, entities: Optional[Iterable[str]] = None, max_pairs: int = 0, progress=None) -> dict:
        """Top R_max relations x top k_max absent tails for ``entities`` (default: every entity).

        ``max_pairs`` bounds the (u, r) queries scored at once ((pairs x E) floats); 0 = automatic.
        """
        Rm, km, Rr = self.R_max, self.k_max, self.Rr
        ids = list(range(self.E)) if entities is None else sorted({self.sc.e2id[e] for e in entities if e in self.sc.e2id})
        n = len(ids)
        max_pairs = max_pairs or int(max(16, min(2048, 2.5e7 // self.E)))
        rel = np.full((n, Rm), -1, dtype=np.int32)
        rel_score = np.full((n, Rm), -np.inf, dtype=np.float32)
        tail = np.full((n, Rm, km), -1, dtype=np.int32)
        tail_score = np.full((n, Rm, km), -np.inf, dtype=np.float32)
        tail_logp = np.full((n, Rm, km), -np.inf, dtype=np.float32)

        U = max(1, max_pairs // self._pairs_per_entity())
        it = range(0, n, U)
        for i in (progress(it) if progress else it):
            out = self._build_chunk(ids[i:i + U])
            c, w = out[0].shape
            rel[i:i + c, :w], rel_score[i:i + c, :w] = out[0], out[1]
            tail[i:i + c, :w], tail_score[i:i + c, :w], tail_logp[i:i + c, :w] = out[2], out[3], out[4]

        # Domain-filtered relations and fully masked tails contribute nothing.
        rel[~np.isfinite(rel_score)] = -1
        tail[rel < 0] = -1
        tail[~np.isfinite(tail_score)] = -1
        self.table = {"ent": np.asarray(ids, dtype=np.int64), "rel": rel, "rel_score": rel_score, "tail": tail,
                      "tail_score": tail_score, "tail_logp": tail_logp}
        self.row = {e: j for j, e in enumerate(ids)}
        return self.table

    def _pairs_per_entity(self) -> int:
        if self.rel_rank == "cooc":
            return self.R_max
        return self.Rr if self.rel_pool is None else min(self.rel_pool, self.Rr)

    def _build_chunk(self, chunk: Sequence[int], allowed: Optional[torch.Tensor] = None):
        """Top relations x top tails for a chunk of entity ids; ``allowed`` (E,) bool restricts tails."""
        Rm, km, Rr = self.R_max, self.k_max, self.Rr
        pool = self._pairs_per_entity()
        cooc = self.cooc_rows(chunk)  # (c, Rr)
        domain = cooc > self.tau if self.signature_filter else torch.ones_like(cooc, dtype=torch.bool)
        if self.rel_rank == "cooc":
            key_extra, r_cand = self.top_relations(cooc, domain, min(Rm, Rr))  # (c, Rm)
        elif pool < Rr:  # KGE rankings over the relations with the highest cooc
            key_extra, r_cand = None, self.top_relations(cooc, domain, pool)[1]
        else:
            key_extra, r_cand = None, torch.arange(Rr, device=self.dev).repeat(len(chunk), 1)
        c, P = r_cand.shape
        u = torch.tensor(list(chunk)).repeat_interleave(P)
        r = r_cand.reshape(-1).cpu()
        sc, lse = self._scores(u, r)
        if allowed is not None:
            sc.masked_fill_(~allowed.to(sc.device).unsqueeze(0), -torch.inf)
        tv = sc.topk(km, dim=1)
        vals = tv.values.view(c, P, km)
        logp = (tv.values - lse).view(c, P, km)
        idx = tv.indices.view(c, P, km)
        dom = domain.gather(1, r_cand)
        if self.rel_rank == "cooc":
            key = key_extra
        elif self.rel_rank == "kge":  # raw score of the best absent tail
            key = vals[:, :, 0]
        elif self.rel_rank == "kge_prob":  # log P(best absent tail | u, r)
            key = logp[:, :, 0]
        else:  # "hybrid": log cooc(u, r) + log P(best absent tail | u, r)
            key = logp[:, :, 0] + torch.log(cooc.gather(1, r_cand).clamp(min=1e-12))
        key = key.masked_fill(~dom, -torch.inf)
        if self.rel_rank == "cooc":  # already in tie-broken order; topk would reshuffle ties
            pos = -torch.arange(P, device=key.device, dtype=key.dtype).expand(c, P)
            top_idx = pos.masked_fill(~torch.isfinite(key), -torch.inf).topk(min(Rm, P), dim=1).indices
        else:
            top_idx = key.topk(min(Rm, P), dim=1).indices
        g = top_idx.unsqueeze(2).expand(-1, -1, km)
        return (r_cand.gather(1, top_idx).cpu().numpy(), key.gather(1, top_idx).cpu().numpy(),
                idx.gather(1, g).cpu().numpy(), vals.gather(1, g).cpu().numpy(), logp.gather(1, g).cpu().numpy())

    @torch.no_grad()
    def local_table(self, u: str, allowed_entities: Iterable[str]) -> Optional[dict]:
        """Like one row of ``build`` but tails are restricted to ``allowed_entities`` (e.g. the
        question's observed subgraph). Relation ranking uses the best *allowed* tail. None if u is unknown."""
        ui = self.sc.e2id.get(u)
        if ui is None:
            return None
        allowed = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
        ids = [self.sc.e2id[e] for e in allowed_entities if e in self.sc.e2id]
        if ids:
            allowed[torch.tensor(ids, device=self.dev)] = True
        rel, rel_score, tail, tail_score, tail_logp = (x[0] for x in self._build_chunk([ui], allowed))
        rel[~np.isfinite(rel_score)] = -1
        tail[rel < 0] = -1
        tail[~np.isfinite(tail_score)] = -1
        return {"rel": rel, "tail": tail, "tail_score": tail_score, "tail_logp": tail_logp}

    def candidates_from(self, u: str, row: Optional[dict], R: int, k: int) -> List[Candidate]:
        """Slice a ``local_table`` row at (R, k)."""
        if row is None:
            return []
        out = []
        for a in range(min(R, len(row["rel"]))):
            ri = int(row["rel"][a])
            if ri < 0:
                continue
            for b in range(min(k, row["tail"].shape[1])):
                vi = int(row["tail"][a, b])
                if vi < 0:
                    break
                out.append(Candidate(u, self.sc.relations[ri], self.sc.entities[vi], float(row["tail_score"][a, b]), a, b,
                                     float(row["tail_logp"][a, b])))
        return out

    # ------------------------------------------------------------------ io
    def config(self) -> dict:
        return {"rel_rank": self.rel_rank, "signature_filter": self.signature_filter, "tau": self.tau,
                "R_max": self.R_max, "k_max": self.k_max, "rel_pool": self.rel_pool,
                "entities_built": int(len(self.table.get("ent", [])))}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self.table)
        with open(path.replace(".npz", ".json"), "w") as f:
            json.dump(self.config(), f, indent=2)

    def load(self, path: str) -> "CandidateGenerator":
        z = np.load(path)
        self.table = {k: z[k] for k in z.files}
        if "ent" not in self.table:  # tables written before subset builds cover every entity
            self.table["ent"] = np.arange(self.table["rel"].shape[0])
        self.row = {int(e): j for j, e in enumerate(self.table["ent"])}
        return self

    # ------------------------------------------------------------------ query
    def candidates(self, u: str, R: int, k: int) -> List[Candidate]:
        """Absent edges (u, r, v) for the top-R relations and top-k tails, canonical ids."""
        j = self.row.get(self.sc.e2id.get(u, -1))
        if j is None:
            return []
        T = self.table
        out = []
        for a in range(min(R, self.R_max)):
            ri = int(T["rel"][j, a])
            if ri < 0:
                continue
            r = self.sc.relations[ri]
            for b in range(min(k, self.k_max)):
                vi = int(T["tail"][j, a, b])
                if vi < 0:
                    break
                lp = float(T["tail_logp"][j, a, b]) if "tail_logp" in T else float("nan")
                out.append(Candidate(u, r, self.sc.entities[vi], float(T["tail_score"][j, a, b]), a, b, lp))
        return out

    def count(self, R: int, k: int) -> Dict[str, int]:
        """Number of candidates per built entity at (R, k)."""
        n = (self.table["tail"][:, :R, :k] >= 0).sum((1, 2))
        return {self.sc.entities[e]: int(c) for e, c in zip(self.table["ent"], n)}

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
        s, _ = self._scores(torch.tensor([ui]), torch.tensor([ri]))
        s = s[0]
        if not torch.isfinite(s[vi]):
            return -1
        return int((s > s[vi]).sum().item())

    def relation_rank(self, u: str, r: str) -> int:
        """0-based rank of r among u's stored relations; -1 if r is outside the top R_max or u was not built."""
        j = self.row.get(self.sc.e2id[u])
        if j is None:
            return -1
        hit = np.where(self.table["rel"][j] == self.sc.r2id[r])[0]
        return int(hit[0]) if len(hit) else -1
