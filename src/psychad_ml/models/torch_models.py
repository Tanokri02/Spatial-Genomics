"""PyTorch models: cell-level MLP and donor-level attention MIL.

Both take sparse CSR log-normalised expression (or a dense input such as PCA / scVI latent) and densify one mini-batch
at a time, so memory stays flat. Early stopping uses held-out *donors* from the training set (never test donors, never
random cells).

Training options (all optional; leaving them out reproduces the original behaviour exactly):
  schedule: onecycle | constant | cosine      onecycle (default) warms up, so its first epoch runs at a tiny LR
  eval_every_steps: 100                       MLP: validate every N optimiser steps instead of once per epoch
  val_metric: loss | donor_auroc              early-stopping criterion; donor_auroc = macro AUROC of donor-averaged
                                              probabilities on the validation donors (what we are actually scored on)
  n_ensemble: 5                               train N members (different seeds AND different validation donors) and
                                              average their probabilities
  MIL only: pooling: global | per_class       per_class = attention pooling within each cell class, then the class
                                              embeddings are concatenated (mirrors the pseudobulk-by-class model)
            cells_per_class: 32               bag = this many cells of each class (per_class pooling)
            val_bags: 4                       bags per validation donor
After fit, `val_proba_` / `val_y_` hold the validation donors' probabilities at the chosen checkpoint (pooled over
ensemble members); the rare-class decision rule is tuned on them.
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


def _scheduler(opt, p: dict, total_steps: int):
    s = p.get("schedule", "onecycle")
    if s == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=max(total_steps, 1))
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1), eta_min=p["lr"] * 0.05)
    if s == "constant":
        return None
    raise ValueError(f"unknown schedule {s}")


def _macro_auroc(y: np.ndarray, P: np.ndarray) -> float:
    from ..evaluate import classification_metrics

    v = classification_metrics(np.asarray(y).astype(int), P, P.shape[1])["macro_auroc"]
    return float(v) if np.isfinite(v) else 0.5


class _Stopper:
    """Tracks the best checkpoint. val_metric 'loss' (lower is better) or 'donor_auroc' (higher is better)."""

    def __init__(self, metric: str, patience: int):
        self.metric, self.patience = metric, patience
        self.best_crit, self.best_state, self.bad = np.inf, None, 0
        self.best_loss, self.best_auroc, self.best_at = np.inf, np.nan, 0

    def update(self, net, at: int, loss: float, auroc: float) -> bool:
        """Returns True when patience is exhausted."""
        crit = loss if self.metric == "loss" else -auroc
        if crit < self.best_crit - 1e-4:
            self.best_crit, self.best_state, self.bad = crit, copy.deepcopy(net.state_dict()), 0
            self.best_loss, self.best_auroc, self.best_at = loss, auroc, at
            return False
        self.bad += 1
        return self.bad >= self.patience


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


def _member_seeds(seed: int, n: int) -> list[int]:
    return [seed] if n == 1 else [seed * 1000 + m for m in range(n)]


def _summarise_members(model, infos: list[dict]) -> None:
    model.best_val_loss_ = float(np.mean([i["best_val_loss"] for i in infos]))
    model.best_epoch_ = int(round(np.mean([i["best_epoch"] for i in infos])))
    model.epochs_run_ = int(round(np.mean([i["epochs_run"] for i in infos])))
    aur = [i["best_val_auroc"] for i in infos if np.isfinite(i["best_val_auroc"])]
    model.train_info_ = {"n_ensemble": len(infos), "best_val_auroc": float(np.mean(aur)) if aur else float("nan"),
                         "best_step_mean": float(np.mean([i["best_step"] for i in infos])),
                         "steps_run_mean": float(np.mean([i["steps_run"] for i in infos]))}


class MLPModel:
    """Cell-level classifier. Weighted CE so donors and classes contribute equally."""

    def __init__(self, params: dict, n_classes: int, seed: int = 0, device: str = "cpu", **_):
        self.p = dict(params)
        self.n_classes, self.seed, self.device = n_classes, seed, device
        self.nets: list[MLPNet] = []
        self.net: MLPNet | None = None

    @property
    def n_params(self) -> int:
        return sum(_n_params(n) for n in self.nets)

    def fit(self, X, y, w=None, groups=None):
        y = np.asarray(y)
        groups = np.asarray(groups) if groups is not None else np.arange(len(y))
        self.nets, infos, vps, vys = [], [], [], []
        for s in _member_seeds(self.seed, int(self.p.get("n_ensemble", 1))):
            net, info, vp, vy = self._fit_one(X, y, groups, s)
            self.nets.append(net)
            infos.append(info)
            if vp is not None:
                vps.append(vp)
                vys.append(vy)
        self.net = self.nets[0]
        _summarise_members(self, infos)
        self.val_proba_ = np.vstack(vps) if vps else None
        self.val_y_ = np.concatenate(vys) if vys else None
        return self

    def _fit_one(self, X, y, groups, seed):
        p, dev = self.p, self.device
        _seed_all(seed)
        donor_label = pd.Series(y, index=groups).groupby(level=0).first()
        val_donors = split_val_donors(donor_label, p.get("val_frac", 0.15), seed)
        is_val = np.isin(groups, list(val_donors)) if val_donors else np.zeros(len(y), bool)
        tr, va = np.where(~is_val)[0], np.where(is_val)[0]
        # re-balance weights within the training part only
        w_tr = make_sample_weights(y[tr], groups[tr])
        w_va = make_sample_weights(y[va], groups[va]) if len(va) else None

        net = MLPNet(X.shape[1], p["hidden"], self.n_classes, p["dropout"], p.get("input_dropout", 0.0)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        bs, epochs = p["batch_size"], p["epochs"]
        steps_per_epoch = int(np.ceil(len(tr) / bs))
        sched = _scheduler(opt, p, epochs * steps_per_epoch)
        every = p.get("eval_every_steps")
        metric = p.get("val_metric", "loss")
        stop = _Stopper(metric, p.get("patience", 5))
        yt = torch.from_numpy(y).long()
        rng = np.random.default_rng(seed)
        step, epochs_run, best_epoch, done = 0, 0, 0, False

        def evaluate(at_epoch: int) -> bool:
            nonlocal best_epoch
            loss, auroc = self._val_eval(net, X, y, groups, va, w_va, bs, need_auroc=(metric == "donor_auroc"))
            out = stop.update(net, step, loss, auroc)
            if stop.bad == 0:                       # this evaluation is the new best checkpoint
                best_epoch = at_epoch
            return out

        for ep in range(epochs):
            epochs_run = ep + 1
            net.train()
            perm = rng.permutation(len(tr))
            for i in range(0, len(perm), bs):
                b = perm[i:i + bs]
                idx = tr[b]
                xb = _dense(X, idx, dev)
                loss = (F.cross_entropy(net(xb), yt[idx].to(dev), reduction="none")
                        * torch.from_numpy(w_tr[b]).to(dev)).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                if sched is not None:
                    sched.step()
                step += 1
                if every and len(va) and step % every == 0:
                    if evaluate(ep + 1):
                        done = True
                        break
                    net.train()
            if done:
                break
            if not every and len(va) and evaluate(ep + 1):
                break
        if stop.best_state is not None:
            net.load_state_dict(stop.best_state)
        vp = vy = None
        if len(va):
            vp, vy = self._val_donor_proba(net, X, y, groups, va, bs)
        info = {"best_val_loss": stop.best_loss, "best_val_auroc": stop.best_auroc, "best_epoch": best_epoch,
                "epochs_run": epochs_run, "best_step": stop.best_at, "steps_run": step}
        return net, info, vp, vy

    @torch.no_grad()
    def _cell_proba(self, net, X, idx, bs) -> np.ndarray:
        net.eval()
        out = [F.softmax(net(_dense(X, idx[i:i + bs], self.device)), 1).cpu().numpy() for i in range(0, len(idx), bs)]
        return np.concatenate(out) if out else np.zeros((0, self.n_classes))

    @torch.no_grad()
    def _val_eval(self, net, X, y, groups, va, w_va, bs, need_auroc: bool) -> tuple[float, float]:
        net.eval()
        tot, probs = 0.0, []
        for i in range(0, len(va), bs):
            idx = va[i:i + bs]
            lg = net(_dense(X, idx, self.device))
            l = F.cross_entropy(lg, torch.from_numpy(y[idx]).long().to(self.device), reduction="none")
            tot += float((l * torch.from_numpy(w_va[i:i + bs]).to(self.device)).sum())
            if need_auroc:
                probs.append(F.softmax(lg, 1).cpu().numpy())
        auroc = np.nan
        if need_auroc:
            from ..evaluate import aggregate_donor

            dp = aggregate_donor(np.concatenate(probs), groups[va])
            dy = pd.Series(y[va], index=groups[va]).groupby(level=0).first().loc[dp.index].values
            auroc = _macro_auroc(dy, dp.values)
        return tot / len(va), auroc

    def _val_donor_proba(self, net, X, y, groups, va, bs):
        from ..evaluate import aggregate_donor

        dp = aggregate_donor(self._cell_proba(net, X, va, bs), groups[va])
        dy = pd.Series(y[va], index=groups[va]).groupby(level=0).first().loc[dp.index].values
        return dp.values, dy

    @torch.no_grad()
    def predict_proba(self, X, bs: int = 4096) -> np.ndarray:
        idx = np.arange(X.shape[0])
        return np.mean([self._cell_proba(n, X, idx, bs) for n in self.nets], axis=0)


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


class ClassMILNet(nn.Module):
    """Cell-class-aware MIL: shared cell encoder; gated attention pooling WITHIN each cell class (class-specific
    attention vector); the C class embeddings (+ a 'class missing' flag each) are concatenated -> donor logits."""

    def __init__(self, in_dim, hidden, attn_dim, n_classes_cell, n_out, dropout, input_dropout):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden, dropout, input_dropout)
        d = self.trunk.out_dim
        self.attn_v = nn.Linear(d, attn_dim)
        self.attn_u = nn.Linear(d, attn_dim)
        self.attn_w = nn.Parameter(torch.randn(n_classes_cell, attn_dim) * 0.05)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(n_classes_cell * (d + 1), n_out)

    def forward(self, bags, mask):  # bags (B, C, K, D), mask (B, C) 1 = class present
        B, C, K, D = bags.shape
        h = self.trunk(bags.reshape(B * C * K, D)).reshape(B, C, K, -1)
        g = torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))           # (B, C, K, a)
        s = torch.einsum("bcka,ca->bck", g, self.attn_w)
        a = torch.softmax(s, dim=2)
        z = (a.unsqueeze(-1) * h).sum(2) * mask.unsqueeze(-1)                    # (B, C, d)
        z = torch.cat([z, (1 - mask).unsqueeze(-1)], dim=-1)
        return self.head(self.drop(z.reshape(B, -1))), a


class MILModel:
    """Donor-level classifier: trained on random bags of cells, predicts one probability vector per donor.

    `extra` (optional, dense n_cells x m) is concatenated to the expression (e.g. cell-class one-hot) so the
    encoder knows which cell class each nucleus belongs to. With pooling=per_class, `extra` must be the cell-class
    one-hot; it then defines the class groups instead of being appended.
    """

    def __init__(self, params: dict, n_classes: int, seed: int = 0, device: str = "cpu", **_):
        self.p = dict(params)
        self.n_classes, self.seed, self.device = n_classes, seed, device
        self.nets: list[nn.Module] = []
        self.net: nn.Module | None = None
        self.per_class = self.p.get("pooling", "global") == "per_class"

    @property
    def n_params(self) -> int:
        return sum(_n_params(n) for n in self.nets)

    @staticmethod
    def _index_by_donor(donors: np.ndarray) -> dict:
        s = pd.Series(np.arange(len(donors))).groupby(np.asarray(donors)).apply(lambda x: x.values)
        return s.to_dict()

    def _index(self, donors, extra) -> dict:
        """global: donor -> cell positions. per_class: donor -> list (one per cell class) of cell positions."""
        if not self.per_class:
            return self._index_by_donor(donors)
        if extra is None:
            raise ValueError("pooling=per_class needs the cell-class one-hot as `extra`")
        cls = np.asarray(extra).argmax(1)
        C = np.asarray(extra).shape[1]
        out = {}
        for d, pos in self._index_by_donor(donors).items():
            out[d] = [pos[cls[pos] == c] for c in range(C)]
        return out

    # ---- bags
    def _make_bag(self, X, extra, cells: np.ndarray, k: int, rng) -> torch.Tensor:
        idx = rng.choice(cells, k, replace=len(cells) < k)
        x = X[idx]
        x = x.toarray() if sp.issparse(x) else np.asarray(x)
        if extra is not None:
            x = np.hstack([x, extra[idx]])
        return torch.from_numpy(x.astype(np.float32, copy=False))

    def _batch(self, X, extra, by_donor, donor_ids, k, rng):
        return torch.stack([self._make_bag(X, extra, by_donor[d], k, rng) for d in donor_ids]).to(self.device)

    def _batch_pc(self, X, by_donor, donor_ids, k, rng):
        """(B, C, k, D) bags with k cells of every class, and the (B, C) presence mask."""
        C = len(next(iter(by_donor.values())))
        idx = np.zeros((len(donor_ids), C, k), dtype=np.int64)
        mask = np.zeros((len(donor_ids), C), dtype=np.float32)
        for j, d in enumerate(donor_ids):
            for c, cells in enumerate(by_donor[d]):
                if len(cells):
                    idx[j, c] = rng.choice(cells, k, replace=len(cells) < k)
                    mask[j, c] = 1.0
        flat = idx.reshape(-1)
        x = X[flat]
        x = x.toarray() if sp.issparse(x) else np.asarray(x)
        x = x.astype(np.float32, copy=False).reshape(len(donor_ids), C, k, -1)
        x = x * mask[:, :, None, None]
        return torch.from_numpy(x).to(self.device), torch.from_numpy(mask).to(self.device)

    def _forward(self, net, X, extra, by_donor, donor_ids, rng):
        if self.per_class:
            xb, m = self._batch_pc(X, by_donor, donor_ids, self.p["cells_per_class"], rng)
            return net(xb, m)[0]
        return net(self._batch(X, extra, by_donor, donor_ids, self.p["bag_size"], rng))[0]

    # ---- training
    def fit(self, X, donors, donor_labels: pd.Series, extra=None):
        """donors: per-cell donor id; donor_labels: Series indexed by donor id with integer labels."""
        self.nets, infos, vps, vys = [], [], [], []
        for s in _member_seeds(self.seed, int(self.p.get("n_ensemble", 1))):
            net, info, vp, vy = self._fit_one(X, donors, donor_labels, extra, s)
            self.nets.append(net)
            infos.append(info)
            if vp is not None:
                vps.append(vp)
                vys.append(vy)
        self.net = self.nets[0]
        _summarise_members(self, infos)
        self.val_proba_ = np.vstack(vps) if vps else None
        self.val_y_ = np.concatenate(vys) if vys else None
        return self

    def _fit_one(self, X, donors, donor_labels, extra, seed):
        p, dev = self.p, self.device
        _seed_all(seed)
        rng = np.random.default_rng(seed)
        by_donor = self._index(donors, extra)
        val = split_val_donors(donor_labels, p.get("val_frac", 0.15), seed)
        tr_labels = donor_labels[[d for d in donor_labels.index if d not in val]]
        va_labels = donor_labels[list(val)] if val else None
        if self.per_class:
            net = ClassMILNet(X.shape[1], p["hidden"], p["attn_dim"], np.asarray(extra).shape[1], self.n_classes,
                              p["dropout"], p.get("input_dropout", 0.0)).to(dev)
            x_extra = None
        else:
            in_dim = X.shape[1] + (extra.shape[1] if extra is not None else 0)
            net = ABMILNet(in_dim, p["hidden"], p["attn_dim"], self.n_classes, p["dropout"],
                           p.get("input_dropout", 0.0)).to(dev)
            x_extra = extra
        opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
        steps, nb = p["steps"], p["batch_bags"]
        sched = _scheduler(opt, p, steps)
        metric = p.get("val_metric", "loss")
        stop = _Stopper(metric, p["patience_evals"])
        # class-balanced donor sampling: pick a class uniformly, then a donor of that class uniformly
        by_class = {c: tr_labels.index[tr_labels.values == c].values for c in np.unique(tr_labels.values)}
        classes = np.array(list(by_class))
        steps_run = 0                                 # for MIL "epochs" count training steps
        for step in range(1, steps + 1):
            steps_run = step
            net.train()
            ds = [rng.choice(by_class[c]) for c in rng.choice(classes, nb)]
            logits = self._forward(net, X, x_extra, by_donor, ds, rng)
            yb = torch.tensor([int(donor_labels[d]) for d in ds], device=dev)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if sched is not None:
                sched.step()
            if va_labels is not None and step % p["eval_every"] == 0:
                loss_v, auroc_v = self._val_eval(net, X, x_extra, by_donor, va_labels)
                if stop.update(net, step, loss_v, auroc_v):
                    break
        if stop.best_state is not None:
            net.load_state_dict(stop.best_state)
        vp = vy = None
        if va_labels is not None:
            rng_v = np.random.default_rng(54321)
            vp = self._predict(net, X, x_extra, by_donor, list(va_labels.index), rng_v, self.p.get("val_bags", 2))
            vy = va_labels.values.astype(int)
        info = {"best_val_loss": stop.best_loss, "best_val_auroc": stop.best_auroc, "best_epoch": stop.best_at,
                "epochs_run": steps_run, "best_step": stop.best_at, "steps_run": steps_run}
        return net, info, vp, vy

    @torch.no_grad()
    def _val_eval(self, net, X, extra, by_donor, va_labels: pd.Series) -> tuple[float, float]:
        rng = np.random.default_rng(12345)  # fixed bags => comparable across evals
        probs = self._predict(net, X, extra, by_donor, list(va_labels.index), rng, n_bags=self.p.get("val_bags", 2))
        y = va_labels.values.astype(int)
        # class-balanced NLL
        nll = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-6, 1))
        loss = float(pd.Series(nll).groupby(y).mean().mean())
        auroc = _macro_auroc(y, probs) if self.p.get("val_metric", "loss") == "donor_auroc" else np.nan
        return loss, auroc

    @torch.no_grad()
    def _predict(self, net, X, extra, by_donor, donor_ids, rng, n_bags) -> np.ndarray:
        net.eval()
        out = np.zeros((len(donor_ids), self.n_classes))
        if self.per_class:                        # batched over donors
            for i in range(0, len(donor_ids), 32):
                chunk = donor_ids[i:i + 32]
                acc = 0
                for _ in range(n_bags):
                    acc = acc + F.softmax(self._forward(net, X, None, by_donor, chunk, rng), 1).cpu().numpy()
                out[i:i + len(chunk)] = acc / n_bags
            return out
        k = self.p["bag_size"]  # same bag size as training; averaging several bags covers more cells
        for j, d in enumerate(donor_ids):
            acc = 0
            for _ in range(n_bags):
                logits, _ = net(self._batch(X, extra, by_donor, [d], k, rng))
                acc = acc + F.softmax(logits, 1).cpu().numpy()[0]
            out[j] = acc / n_bags
        return out

    def predict_donor_proba(self, X, donors, extra=None) -> pd.DataFrame:
        by_donor = self._index(donors, extra)
        x_extra = None if self.per_class else extra
        ids = sorted(by_donor)
        P = []
        for m, net in enumerate(self.nets):
            rng = np.random.default_rng(self.seed + 999 + m)
            P.append(self._predict(net, X, x_extra, by_donor, ids, rng, self.p.get("infer_bags", 8)))
        return pd.DataFrame(np.mean(P, axis=0), index=ids)
