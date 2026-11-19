"""ComplEx knowledge-graph embeddings for the candidate generator and the plausibility prior.

The model is trained on the *observed* (incomplete) graph only: BRINK's removed
triples and the test questions are never seen, so candidate recall and pi(e)
are not inflated by leakage (plan section 6.4).

Inverse edges are explicit relations ``~r`` (reciprocal training), matching the
inverse-augmented graph the KG-Trie walks (``cc_gcd.data.brink``). Every
candidate query is then a tail query (u, r, ?) with r in R and ~R.

Training follows the standard 1-vs-all recipe of Lacroix, Usunier & Obozinski
(ICML 2018, "Canonical Tensor Decomposition for Knowledge Base Completion"):
cross-entropy over all tails, N3 regularization on the factors of each query,
Adagrad. It is implemented here in plain PyTorch (about 40 lines) because the
equivalent PyKEEN 1.11 pipeline (LCWA + Adam) converged far more slowly on
FB15k-237 (filtered MRR 0.06 after 20 epochs vs 0.2-0.3 expected).

Early stopping uses a random holdout of observed triples (MCAR), not BRINK's
rule-based removals, because BRINK is a test bed and not a calibration source
(plan section 6.2). The holdout is split on canonical triples before inverse
augmentation, so (h, r, t) and (t, ~r, h) always land in the same split.

Embeddings are exported to ``embeddings.npz``; score(h, r, t) = Re(<e_h, w_r, conj(e_t)>).
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

Triple = Tuple[str, str, str]
INVERSE_PREFIX = "~"


def inverse(r: str) -> str:
    return r[len(INVERSE_PREFIX):] if r.startswith(INVERSE_PREFIX) else INVERSE_PREFIX + r


def augment(triples: Sequence[Triple]) -> List[Triple]:
    return [x for h, r, t in triples for x in ((h, r, t), (t, INVERSE_PREFIX + r, h))]


class _ComplEx(torch.nn.Module):
    def __init__(self, n_ent: int, n_rel: int, dim: int, init_scale: float = 1e-3, sparse: bool = False):
        super().__init__()
        # Real and imaginary parts stored side by side: [re | im].
        self.ent = torch.nn.Embedding(n_ent, 2 * dim, sparse=sparse)
        self.rel = torch.nn.Embedding(n_rel, 2 * dim, sparse=sparse)
        self.ent.weight.data *= init_scale
        self.rel.weight.data *= init_scale
        self.dim = dim

    def forward(self, h: torch.Tensor, r: torch.Tensor):
        d = self.dim
        lhs, rel = self.ent(h), self.rel(r)
        lr, li, rr, ri = lhs[:, :d], lhs[:, d:], rel[:, :d], rel[:, d:]
        qr, qi = lr * rr - li * ri, lr * ri + li * rr
        E = self.ent.weight
        scores = qr @ E[:, :d].T + qi @ E[:, d:].T
        return scores, (torch.sqrt(lr ** 2 + li ** 2), torch.sqrt(rr ** 2 + ri ** 2))

    def score_candidates(self, h: torch.Tensor, r: torch.Tensor, cand: torch.Tensor):
        """Scores against a candidate subset of tails (sampled softmax); gradients stay sparse."""
        d = self.dim
        lhs, rel, C = self.ent(h), self.rel(r), self.ent(cand)
        lr, li, rr, ri = lhs[:, :d], lhs[:, d:], rel[:, :d], rel[:, d:]
        qr, qi = lr * rr - li * ri, lr * ri + li * rr
        scores = qr @ C[:, :d].T + qi @ C[:, d:].T
        return scores, (torch.sqrt(lr ** 2 + li ** 2), torch.sqrt(rr ** 2 + ri ** 2))


@torch.no_grad()
def _filtered_ranks(model: _ComplEx, triples: np.ndarray, known: Dict[Tuple[int, int], List[int]],
                    device: str, batch: int = 0) -> np.ndarray:
    batch = batch or int(min(512, max(16, 5e7 // model.ent.num_embeddings)))  # (batch x E) fp32 scores <= 200 MB
    ranks = []
    for i in range(0, len(triples), batch):
        b = torch.as_tensor(triples[i:i + batch], device=device)
        s, _ = model(b[:, 0], b[:, 1])
        tgt = s[torch.arange(len(b)), b[:, 2]].clone()
        rows, cols = [], []
        for j, (h, r, _) in enumerate(triples[i:i + batch]):
            ks = known.get((int(h), int(r)), [])
            rows += [j] * len(ks)
            cols += ks
        if rows:
            s[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = -torch.inf
        ranks.append((1 + (s > tgt.unsqueeze(1)).sum(1)).cpu().numpy())
    return np.concatenate(ranks)


def train_complex(triples: Sequence[Triple], out_dir: str, *, dim: int = 256, epochs: int = 100, lr: float = 0.1,
                  batch_size: int = 512, reg_weight: float = 5e-2, valid_frac: float = 0.02, patience: int = 3,
                  eval_every: int = 5, seed: int = 0, device: str = "cuda", amp: bool = False,
                  neg_sample: int = 0, log=print) -> dict:
    """Train reciprocal ComplEx-N3 on observed triples; keep the best epoch by holdout filtered MRR.

    ``amp`` runs the score matmul in bf16 (loss and N3 stay fp32). ``neg_sample`` > 0 replaces
    1-vs-all with a sampled softmax: each batch scores its true tails plus ``neg_sample`` shared
    uniform entities, and embeddings get sparse gradients. With uniform sampling the softmax
    correction is a constant, so the objective keeps its form. Needed when the full entity table's
    dense gradient and score matrices do not fit next to the desktop on the 8 GB RTX 3060 Ti
    (IKGQA-WebQSP: 599k entities). Evaluation always ranks against every entity.
    """
    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    canon = sorted(set(map(tuple, triples)))
    rng = random.Random(seed)
    valid_idx = set(rng.sample(range(len(canon)), int(valid_frac * len(canon))))
    train_c = [t for i, t in enumerate(canon) if i not in valid_idx]
    valid_c = [t for i, t in enumerate(canon) if i in valid_idx]

    # Vocabularies come from *all* observed triples so every observed entity has an embedding.
    entities = sorted({e for h, _, t in canon for e in (h, t)})
    relations = sorted({r for _, r, _ in augment(canon)})
    e2id = {e: i for i, e in enumerate(entities)}
    r2id = {r: i for i, r in enumerate(relations)}

    def ids(rows):
        return np.array([(e2id[h], r2id[r], e2id[t]) for h, r, t in augment(rows)], dtype=np.int64)

    tr, va = ids(train_c), ids(valid_c)
    known: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for h, r, t in np.concatenate([tr, va]):
        known[(int(h), int(r))].append(int(t))

    model = _ComplEx(len(entities), len(relations), dim, sparse=neg_sample > 0).to(device)
    E_n = len(entities)
    opt = torch.optim.Adagrad(model.parameters(), lr=lr)
    tr_t = torch.as_tensor(tr, device=device)
    best, best_state, best_epoch, bad, t0, history = -1.0, None, 0, 0, time.time(), []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_t), device=device)
        total = 0.0
        for i in range(0, len(perm), batch_size):
            b = tr_t[perm[i:i + batch_size]]
            if neg_sample:
                cand = torch.cat([b[:, 2], torch.randint(0, E_n, (neg_sample,), device=device)])
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    scores, (fl, fr) = model.score_candidates(b[:, 0], b[:, 1], cand)
                scores = scores.float()
                # Row j's target sits at column j; any other column holding the same entity is not a negative.
                dup = cand.unsqueeze(0) == b[:, 2].unsqueeze(1)
                dup[torch.arange(len(b), device=device), torch.arange(len(b), device=device)] = False
                scores = scores.masked_fill(dup, -torch.inf)
                target = torch.arange(len(b), device=device)
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    scores, (fl, fr) = model(b[:, 0], b[:, 1])
                scores = scores.float()
                target = b[:, 2]
            ft = model.ent(b[:, 2])
            ft = torch.sqrt(ft[:, :dim] ** 2 + ft[:, dim:] ** 2)
            loss = torch.nn.functional.cross_entropy(scores, target)
            reg = reg_weight * (fl ** 3 + fr ** 3 + ft ** 3).sum() / len(b)
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            total += loss.item() * len(b)
        if epoch % eval_every == 0 or epoch == epochs:
            model.eval()
            mrr = float(np.mean(1.0 / _filtered_ranks(model, va, known, device)))
            history.append({"epoch": epoch, "train_ce": round(total / len(tr), 4), "valid_mrr": round(mrr, 4)})
            log(f"epoch {epoch}: train CE {total / len(tr):.4f}, holdout filtered MRR {mrr:.4f}")
            if mrr > best + 1e-4:
                best, best_epoch, bad = mrr, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    ranks = _filtered_ranks(model, va, known, device)

    E = model.ent.weight.detach().cpu().numpy()
    Rw = model.rel.weight.detach().cpu().numpy()
    np.savez(os.path.join(out_dir, "embeddings.npz"), ent_re=E[:, :dim], ent_im=E[:, dim:],
             rel_re=Rw[:, :dim], rel_im=Rw[:, dim:])
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump({"entities": entities, "relations": relations}, f)
    metrics = {
        "model": "ComplEx-N3 (reciprocal, " + (f"sampled softmax {neg_sample}" if neg_sample else "1-vs-all") + " CE, Adagrad)", "dim": dim, "amp_bf16": amp, "neg_sample": neg_sample, "lr": lr, "batch_size": batch_size,
        "reg_weight_n3": reg_weight, "seed": seed, "epochs_max": epochs, "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 1),
        "train_triples_canonical": len(train_c), "valid_triples_canonical": len(valid_c),
        "entities": len(entities), "relations_with_inverse": len(relations),
        # Filtered tail ranks on the MCAR holdout; each canonical triple is scored in both directions.
        "valid_mrr": round(float(np.mean(1.0 / ranks)), 4),
        "valid_hits@1": round(float(np.mean(ranks <= 1)), 4),
        "valid_hits@10": round(float(np.mean(ranks <= 10)), 4),
        "history": history,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


class ComplExScorer:
    """Scores (u, r, ?) tail queries from exported ComplEx embeddings."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        z = np.load(os.path.join(model_dir, "embeddings.npz"))
        with open(os.path.join(model_dir, "vocab.json")) as f:
            vocab = json.load(f)
        self.device = device
        self.entities: List[str] = vocab["entities"]
        self.relations: List[str] = vocab["relations"]
        self.e2id: Dict[str, int] = {e: i for i, e in enumerate(self.entities)}
        self.r2id: Dict[str, int] = {r: i for i, r in enumerate(self.relations)}
        self.ent = torch.complex(torch.from_numpy(z["ent_re"]), torch.from_numpy(z["ent_im"])).to(device)
        self.rel = torch.complex(torch.from_numpy(z["rel_re"]), torch.from_numpy(z["rel_im"])).to(device)

    @torch.no_grad()
    def score_tails(self, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """(B,) head ids, (B,) relation ids -> (B, E) real scores for every tail."""
        q = self.ent[h] * self.rel[r]
        return (q @ self.ent.conj().T).real

    @torch.no_grad()
    def score(self, h: Sequence[int], r: Sequence[int], t: Sequence[int]) -> np.ndarray:
        h, r, t = (torch.as_tensor(x, device=self.device) for x in (h, r, t))
        return (self.ent[h] * self.rel[r] * self.ent[t].conj()).sum(-1).real.cpu().numpy()
