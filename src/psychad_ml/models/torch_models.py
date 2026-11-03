"""PyTorch models: cell-level MLP and donor-level attention MIL.

Both take sparse CSR log-normalised expression and densify one mini-batch at a time, so memory stays flat.
Early stopping uses held-out *donors* from the training set (never test donors, never random cells).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError("torch is required for MLP/MIL models: pip install torch") from e

from .weights import make_sample_weights


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dense(X, idx, device) -> torch.Tensor:
    rows = X[idx]
    rows = rows.toarray() if sp.issparse(rows) else np.asarray(rows)
    return torch.from_numpy(rows.astype(np.float32, copy=False)).to(device)


def split_val_donors(labels_by_donor: pd.Series, frac: float, seed: int) -> set:
    """Stratified (by label) random donor hold-out. Falls back to no val for classes with <3 donors."""
    rng = np.random.default_rng(seed)
    val = set()
    for _, donors in labels_by_donor.groupby(labels_by_donor):
        ids = np.array(donors.index)
        if len(ids) >= 3:
            val |= set(rng.choice(ids, max(1, int(round(frac * len(ids)))), replace=False))
    return val


class Trunk(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], dropout: float, input_dropout: float):
        super().__init__()
        layers: list[nn.Module] = [nn.Dropout(input_dropout)] if input_dropout > 0 else []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x):
        return self.net(x)


class MLPNet(nn.Module):
    def __init__(self, in_dim, hidden, n_out, dropout, input_dropout):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden, dropout, input_dropout)
        self.head = nn.Linear(self.trunk.out_dim, n_out)

    def forward(self, x):
        return self.head(self.trunk(x))


def _n_params(m: nn.Module) -> int:
    return int(sum(p.numel() for p in m.parameters() if p.requires_grad))


class MLPModel:
    """Cell-level classifier. Weighted CE so donors and classes contribute equally."""

    def __init__(self, params: dict, n_classes: int, seed: int = 0, device: str = "cpu", **_):
        self.p = dict(params)
        self.n_classes, self.seed, self.device = n_classes, seed, device
        self.net: MLPNet | None = None

    @property
    def n_params(self) -> int:
        return _n_params(self.net) if self.net is not None else 0

    def fit(self, X, y, w=None, groups=None):
        p, dev = self.p, self.device
        _seed_all(self.seed)
        y = np.asarray(y)
        groups = np.asarray(groups) if groups is not None else np.arange(len(y))
        if w is None:
            w = make_sample_weights(y, groups)
        donor_label = pd.Series(y, index=groups).groupby(level=0).first()
        val_donors = split_val_donors(donor_label, p.get("val_frac", 0.15), self.seed)
        is_val = np.isin(groups, list(val_donors)) if val_donors else np.zeros(len(y), bool)
        tr, va = np.where(~is_val)[0], np.where(is_val)[0]
        # re-balance weights within the training part only
        w_tr = make_sample_weights(y[tr], groups[tr])
        w_va = make_sample_weights(y[va], groups[va]) if len(va) else None

        self.net = MLPNet(X.shape[1], p["hidden"], self.n_classes, p["dropout"], p.get("input_dropout", 0.0)).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        bs, epochs = p["batch_size"], p["epochs"]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=epochs * int(np.ceil(len(tr) / bs)))
        yt = torch.from_numpy(y).long()
        best, best_state, bad = np.inf, None, 0
        rng = np.random.default_rng(self.seed)
        for ep in range(epochs):
            self.net.train()
            perm = rng.permutation(len(tr))
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                idx = tr[b]
                xb = _dense(X, idx, dev)
                loss = (F.cross_entropy(self.net(xb), yt[idx].to(dev), reduction="none")
                        * torch.from_numpy(w_tr[b]).to(dev)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
                sched.step()
            if len(va):
                v = self._val_loss(X, y, va, w_va, bs)
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= p.get("patience", 5):
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    @torch.no_grad()
    def _val_loss(self, X, y, va, w_va, bs) -> float:
        self.net.eval()
        tot = 0.0
        for i in range(0, len(va), bs):
            idx = va[i:i + bs]
            lg = self.net(_dense(X, idx, self.device))
            l = F.cross_entropy(lg, torch.from_numpy(y[idx]).long().to(self.device), reduction="none")
            tot += float((l * torch.from_numpy(w_va[i:i + bs]).to(self.device)).sum())
        return tot / len(va)

    @torch.no_grad()
    def predict_proba(self, X, bs: int = 4096) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, X.shape[0], bs):
            out.append(F.softmax(self.net(_dense(X, np.arange(i, min(i + bs, X.shape[0])), self.device)), 1).cpu().numpy())
        return np.concatenate(out)


# --------------------------------------------------------------------------------------------------
# Attention-based multiple-instance learning (donor = bag of cells)
# --------------------------------------------------------------------------------------------------
class ABMILNet(nn.Module):
    """Gated-attention MIL (Ilse et al. 2018): cell encoder -> gated attention pooling -> donor logits."""

    def __init__(self, in_dim, hidden, attn_dim, n_out, dropout, input_dropout):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden, dropout, input_dropout)
        d = self.trunk.out_dim
        self.attn_v = nn.Linear(d, attn_dim)
        self.attn_u = nn.Linear(d, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.head = nn.Linear(d, n_out)

    def forward(self, bags):  # bags: (B, K, D)
        B, K, D = bags.shape
        h = self.trunk(bags.reshape(B * K, D)).reshape(B, K, -1)
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)  # (B, K)
        a = torch.softmax(a, dim=1)
        z = (a.unsqueeze(-1) * h).sum(1)
        return self.head(z), a


class MILModel:
    """Donor-level classifier: trained on random bags of cells, predicts one probability vector per donor.

    `extra` (optional, dense n_cells x m) is concatenated to the expression (e.g. cell-class one-hot) so the
    encoder knows which cell class each nucleus belongs to.
    """

    def __init__(self, params: dict, n_classes: int, seed: int = 0, device: str = "cpu", **_):
        self.p = dict(params)
        self.n_classes, self.seed, self.device = n_classes, seed, device
        self.net: ABMILNet | None = None

    @property
    def n_params(self) -> int:
        return _n_params(self.net) if self.net is not None else 0

    @staticmethod
    def _index_by_donor(donors: np.ndarray) -> dict:
        s = pd.Series(np.arange(len(donors))).groupby(np.asarray(donors)).apply(lambda x: x.values)
        return s.to_dict()

    def _make_bag(self, X, extra, cells: np.ndarray, k: int, rng) -> torch.Tensor:
        idx = rng.choice(cells, k, replace=len(cells) < k)
        x = X[idx]
        x = x.toarray() if sp.issparse(x) else np.asarray(x)
        if extra is not None:
            x = np.hstack([x, extra[idx]])
        return torch.from_numpy(x.astype(np.float32, copy=False))

    def _batch(self, X, extra, by_donor, donor_ids, k, rng):
        return torch.stack([self._make_bag(X, extra, by_donor[d], k, rng) for d in donor_ids]).to(self.device)

    def fit(self, X, donors, donor_labels: pd.Series, extra=None):
        """donors: per-cell donor id; donor_labels: Series indexed by donor id with integer labels."""
        p, dev = self.p, self.device
        _seed_all(self.seed)
        rng = np.random.default_rng(self.seed)
        by_donor = self._index_by_donor(donors)
        val = split_val_donors(donor_labels, p.get("val_frac", 0.15), self.seed)
        tr_labels = donor_labels[[d for d in donor_labels.index if d not in val]]
        va_labels = donor_labels[list(val)] if val else None
        in_dim = X.shape[1] + (extra.shape[1] if extra is not None else 0)
        self.net = ABMILNet(in_dim, p["hidden"], p["attn_dim"], self.n_classes, p["dropout"],
                            p.get("input_dropout", 0.0)).to(dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        steps, k, nb = p["steps"], p["bag_size"], p["batch_bags"]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=steps)
        # class-balanced donor sampling: pick a class uniformly, then a donor of that class uniformly
        by_class = {c: tr_labels.index[tr_labels.values == c].values for c in np.unique(tr_labels.values)}
        classes = np.array(list(by_class))
        best, best_state, bad = np.inf, None, 0
        for step in range(1, steps + 1):
            self.net.train()
            ds = [rng.choice(by_class[c]) for c in rng.choice(classes, nb)]
            xb = self._batch(X, extra, by_donor, ds, k, rng)
            yb = torch.tensor([int(donor_labels[d]) for d in ds], device=dev)
            logits, _ = self.net(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()
            sched.step()
            if va_labels is not None and step % p["eval_every"] == 0:
                v = self._val_loss(X, extra, by_donor, va_labels)
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(self.net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= p["patience_evals"]:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.best_val_loss_ = best
        return self

    @torch.no_grad()
    def _val_loss(self, X, extra, by_donor, va_labels: pd.Series) -> float:
        self.net.eval()
        rng = np.random.default_rng(12345)  # fixed bags => comparable across evals
        probs = self._predict(X, extra, by_donor, list(va_labels.index), rng, n_bags=2)
        y = va_labels.values.astype(int)
        # class-balanced NLL
        nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-6, 1))
        return float(pd.Series(nll).groupby(y).mean().mean())

    @torch.no_grad()
    def _predict(self, X, extra, by_donor, donor_ids, rng, n_bags) -> np.ndarray:
        self.net.eval()
        k = self.p["bag_size"]  # same bag size as training; averaging several bags covers more cells
        out = np.zeros((len(donor_ids), self.n_classes))
        for j, d in enumerate(donor_ids):
            acc = 0
            for _ in range(n_bags):
                logits, _ = self.net(self._batch(X, extra, by_donor, [d], k, rng))
                acc = acc + F.softmax(logits, 1).cpu().numpy()[0]
            out[j] = acc / n_bags
        return out

    def predict_donor_proba(self, X, donors, extra=None) -> pd.DataFrame:
        by_donor = self._index_by_donor(donors)
        ids = sorted(by_donor)
        rng = np.random.default_rng(self.seed + 999)
        return pd.DataFrame(self._predict(X, extra, by_donor, ids, rng, self.p.get("infer_bags", 8)), index=ids)
