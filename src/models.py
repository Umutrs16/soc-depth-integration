"""Discrete baselines and the continuous-depth MCU-SoilNet.

The network predicts a non-negative carbon density q(z) in kg m^-2 cm^-1.
Layer stocks are 8-point Gauss–Legendre integrals between the observed
boundaries. There is no gravel-fraction input in V10, so the primary
model does not claim a C–BD–CF product closure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from torch import nn
from xgboost import XGBRegressor

NUMERIC_FEATURES = [
    "elevation",
    "slope",
    "aspect_sin",
    "aspect_cos",
    "tpi_2km",
    "relief_2km",
    "mat_c",
    "tseason_c",
    "trange_c",
    "map_mm",
    "pet_mm",
    "soil_vwc",
    "ndvi_p10",
    "ndvi_p50",
    "ndvi_p90",
    "ndvi_std",
]
OPTICAL_FEATURES = ["ndvi_p10", "ndvi_p50", "ndvi_p90", "ndvi_std"]
WC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
DEPTH_FEATURES = ["top_cm", "bottom_cm", "mid_cm", "thickness_cm"]
GAUSS_N = 8


@dataclass
class ModelSpec:
    name: str
    kind: str
    smooth: float = 0.0
    aux: float = 0.0
    closure: float = 0.0
    use_weights: bool = False
    drop_optical: bool = False
    seeds: tuple[int, ...] = (42,)
    epochs: int = 40
    patience: int = 8
    n_freq: int = 8
    gauss_n: int = 8
    smooth_nodes: int = 8
    cumulative_dz: float = 0.0
    z_max: float = 800.0


SPECS = {
    "rf": ModelSpec("rf", "tree"),
    "xgb": ModelSpec("xgb", "tree"),
    "hgb": ModelSpec("hgb", "tree"),
    "mlp_integral": ModelSpec("mlp_integral", "mcu", epochs=40, patience=8),
    "mcu": ModelSpec("mcu", "mcu", smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=50, patience=8),
    "mcu_equal_weight": ModelSpec("mcu_equal_weight", "mcu", smooth=0.02, aux=0.05, epochs=40, patience=8),
    "mcu_closure": ModelSpec("mcu_closure", "mcu", smooth=0.02, aux=0.05, closure=0.1, use_weights=True, epochs=40, patience=8),
    "mcu_nors": ModelSpec("mcu_nors", "mcu", smooth=0.02, aux=0.05, use_weights=True, drop_optical=True, epochs=40, patience=8),
}


def feature_columns(spec: ModelSpec) -> list[str]:
    cols = [c for c in NUMERIC_FEATURES if not (spec.drop_optical and c in OPTICAL_FEATURES)]
    return cols


def design_matrix(frame, spec: ModelSpec, medians: dict[str, float] | None = None, for_tree: bool = False):
    data = frame.copy()
    data["mid_cm"] = (data["top_cm"] + data["bottom_cm"]) / 2.0
    cols = feature_columns(spec)
    if medians is None:
        medians = {c: float(data[c].median()) if data[c].notna().any() else 0.0 for c in cols}
    x_num = np.column_stack([data[c].fillna(medians[c]).to_numpy(dtype=float) for c in cols])
    wc = data["worldcover"].fillna(-1).round().astype(int).to_numpy()
    wc_oh = np.column_stack([(wc == code).astype(float) for code in WC_CLASSES])
    parts = [x_num, wc_oh]
    names = cols + [f"wc_{code}" for code in WC_CLASSES]
    if for_tree:
        depth = np.column_stack([data[c].to_numpy(dtype=float) for c in DEPTH_FEATURES])
        parts.append(depth)
        names.extend(DEPTH_FEATURES)
    x = np.concatenate(parts, axis=1)
    return x, medians, names


def make_tree(spec: ModelSpec, seed: int = 42):
    if spec.name == "rf":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=22,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        )
    if spec.name == "xgb":
        return XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    if spec.name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=seed,
        )
    raise KeyError(spec.name)


class MCUSoilNet(nn.Module):
    def __init__(self, n_features: int, n_freq: int = 8, gauss_n: int = 8, cumulative_dz: float = 0.0, z_max: float = 800.0):
        super().__init__()
        self.n_freq = int(n_freq)
        self.cumulative_dz = float(cumulative_dz)
        self.z_max = float(z_max)
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(64 + 2 * self.n_freq + 1, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.q_head = nn.Linear(128, 1)
        self.soc_head = nn.Linear(128, 1)
        self.bd_head = nn.Linear(128, 1)
        nn.init.constant_(self.q_head.bias, -1.2)
        nodes, weights = leggauss(int(gauss_n))
        self.register_buffer("gauss_nodes", torch.tensor(nodes, dtype=torch.float32))
        self.register_buffer("gauss_weights", torch.tensor(weights, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def fourier(self, z: torch.Tensor) -> torch.Tensor:
        zn = z / 200.0
        bands = []
        for k in range(self.n_freq):
            freq = (2.0**k) * math.pi * zn
            bands.append(torch.sin(freq))
            bands.append(torch.cos(freq))
        return torch.stack(bands, dim=-1)

    def hidden_at(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ff = self.fourier(z)
        zn = (z / 200.0).unsqueeze(-1)
        return self.trunk(torch.cat([h, ff, zn], dim=-1))

    def q_at(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.q_head(self.hidden_at(h, z)).squeeze(-1))

    def _cumulative_stock(self, h: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        dz = self.cumulative_dz
        n_bins = int(round(self.z_max / dz))
        z_mid = (torch.arange(n_bins, device=h.device, dtype=torch.float32) + 0.5) * dz
        h_exp = h[:, None, :].expand(-1, n_bins, -1)
        q = self.q_at(h_exp.reshape(-1, h.shape[-1]), z_mid.repeat(h.shape[0])).reshape(h.shape[0], n_bins)
        edges = torch.cat(
            [torch.zeros(h.shape[0], 1, device=h.device), torch.cumsum(q * dz, dim=1)],
            dim=1,
        )

        def at(z: torch.Tensor) -> torch.Tensor:
            pos = (z / dz).clamp(0, n_bins)
            i0 = pos.floor().to(torch.long)
            at_end = i0 >= n_bins
            i0 = torch.where(at_end, torch.full_like(i0, n_bins - 1), i0)
            i1 = (i0 + 1).clamp(max=n_bins)
            weight = torch.where(at_end, torch.ones_like(pos), (pos - i0.to(pos.dtype)).clamp(0, 1))
            batch = torch.arange(h.shape[0], device=h.device)[:, None].expand_as(i0)
            return edges[batch, i0] * (1.0 - weight) + edges[batch, i1] * weight

        return at(bottom) - at(top)

    def integrate(self, h: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor) -> torch.Tensor:
        # h: [B, H], top/bottom: [B, L]
        if self.cumulative_dz > 0:
            return self._cumulative_stock(h, top, bottom)
        span = (bottom - top).clamp_min(0)
        z = 0.5 * span.unsqueeze(-1) * self.gauss_nodes + 0.5 * (top + bottom).unsqueeze(-1)
        h_exp = h[:, None, None, :].expand(-1, z.shape[1], z.shape[2], -1)
        q = self.q_at(h_exp.reshape(-1, h.shape[-1]), z.reshape(-1)).reshape_as(z)
        w = 0.5 * span.unsqueeze(-1) * self.gauss_weights
        return (q * w).sum(dim=-1)

    def aux_at_mid(self, h: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor):
        mid = 0.5 * (top + bottom)
        h_exp = h[:, None, :].expand(-1, mid.shape[1], -1)
        hidden = self.hidden_at(h_exp.reshape(-1, h.shape[-1]), mid.reshape(-1)).reshape(
            mid.shape[0], mid.shape[1], -1
        )
        soc = torch.nn.functional.softplus(self.soc_head(hidden).squeeze(-1))
        bd = torch.nn.functional.softplus(self.bd_head(hidden).squeeze(-1))
        q = self.q_at(h_exp.reshape(-1, h.shape[-1]), mid.reshape(-1)).reshape_as(mid)
        return q, soc, bd


def build_profile_cache(frame, x_by_profile):
    cache = {}
    max_layers = 1
    for pid, rows in frame.groupby("profile_id", sort=False):
        key = str(pid)
        n = int(len(rows))
        max_layers = max(max_layers, n)
        cache[key] = {
            "x": np.asarray(x_by_profile[key], dtype=np.float32),
            "top": rows["top_cm"].to_numpy(np.float32),
            "bot": rows["bottom_cm"].to_numpy(np.float32),
            "socd": rows["socd_kgm2"].fillna(0).to_numpy(np.float32),
            "soc": rows["soc_gkg"].fillna(0).to_numpy(np.float32),
            "bd": rows["bd_gcm3"].fillna(0).to_numpy(np.float32),
            "w": rows["train_weight"].to_numpy(np.float32),
            "closure": rows["closure_ok"].astype(np.float32).to_numpy(),
            "soc_m": rows["soc_observed"].astype(np.float32).to_numpy(),
            "bd_m": rows["bd_observed"].astype(np.float32).to_numpy(),
            "n": n,
        }
    return cache, max_layers


def _batch_from_cache(cache, ids, max_layers: int, device):
    xs, tops, bots, socd, soc, bd, mask, w, closure, soc_m, bd_m = [], [], [], [], [], [], [], [], [], [], []
    for pid in ids:
        item = cache[str(pid)]
        n = item["n"]
        pad = max_layers - n

        def take(values):
            if pad:
                return np.concatenate([values, np.zeros(pad, dtype=np.float32)])
            return values

        xs.append(item["x"])
        tops.append(take(item["top"]))
        bots.append(take(item["bot"]))
        socd.append(take(item["socd"]))
        soc.append(take(item["soc"]))
        bd.append(take(item["bd"]))
        layer_mask = np.zeros(max_layers, dtype=np.float32)
        layer_mask[:n] = 1.0
        mask.append(layer_mask)
        w.append(take(item["w"]))
        closure.append(take(item["closure"]))
        soc_m.append(take(item["soc_m"]))
        bd_m.append(take(item["bd_m"]))
    tensors = [xs, tops, bots, socd, soc, bd, mask, w, closure, soc_m, bd_m]
    return [torch.tensor(np.stack(v), dtype=torch.float32, device=device) for v in tensors]


def _batch_tensors(frame, ids, x_lookup, max_layers: int, device):
    xs, tops, bots, socd, soc, bd, mask, w, closure, soc_m, bd_m = [], [], [], [], [], [], [], [], [], [], []
    for pid in ids:
        rows = frame.loc[frame["profile_id"] == pid]
        n = min(len(rows), max_layers)
        rows = rows.iloc[:n]
        pad = max_layers - n
        def pad_array(values):
            arr = np.asarray(values, dtype=np.float32)
            if pad:
                arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
            return arr

        xs.append(x_lookup[pid])
        tops.append(pad_array(rows["top_cm"]))
        bots.append(pad_array(rows["bottom_cm"]))
        socd.append(pad_array(rows["socd_kgm2"].fillna(0)))
        soc.append(pad_array(rows["soc_gkg"].fillna(0)))
        bd.append(pad_array(rows["bd_gcm3"].fillna(0)))
        layer_mask = np.zeros(max_layers, dtype=np.float32)
        layer_mask[:n] = 1.0
        mask.append(layer_mask)
        weight = np.zeros(max_layers, dtype=np.float32)
        weight[:n] = rows["train_weight"].to_numpy(dtype=np.float32)
        w.append(weight)
        clos = np.zeros(max_layers, dtype=np.float32)
        clos[:n] = rows["closure_ok"].astype(float).to_numpy(dtype=np.float32)
        closure.append(clos)
        sm = np.zeros(max_layers, dtype=np.float32)
        sm[:n] = rows["soc_observed"].astype(float).to_numpy(dtype=np.float32)
        soc_m.append(sm)
        bm = np.zeros(max_layers, dtype=np.float32)
        bm[:n] = rows["bd_observed"].astype(float).to_numpy(dtype=np.float32)
        bd_m.append(bm)
    tensors = [xs, tops, bots, socd, soc, bd, mask, w, closure, soc_m, bd_m]
    return [torch.tensor(np.stack(v), dtype=torch.float32, device=device) for v in tensors]


def train_mcu(frame, x_by_profile, spec: ModelSpec, seed: int, fit_ids, val_ids):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = len(next(iter(x_by_profile.values())))
    model = MCUSoilNet(
        n_features,
        n_freq=spec.n_freq,
        gauss_n=spec.gauss_n,
        cumulative_dz=spec.cumulative_dz,
        z_max=spec.z_max,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=spec.epochs)
    cache, max_layers = build_profile_cache(frame, x_by_profile)
    fit_ids = np.asarray(list(fit_ids))
    val_ids = np.asarray(list(val_ids))
    best_state = None
    best_mae = float("inf")
    wait = 0
    rng = np.random.default_rng(seed)
    for _epoch in range(spec.epochs):
        model.train()
        order = rng.permutation(fit_ids)
        for start in range(0, len(order), 48):
            batch_ids = order[start : start + 48]
            x, top, bot, socd, soc, bd, mask, weight, closure, soc_m, bd_m = _batch_from_cache(
                cache, batch_ids, max_layers, device
            )
            if not spec.use_weights:
                weight = torch.ones_like(weight)
            h = model.encode(x)
            pred = model.integrate(h, top, bot)
            diff = pred - socd
            huber = torch.where(diff.abs() <= 2.0, 0.5 * diff**2, 2.0 * (diff.abs() - 1.0))
            denom = (mask * weight).sum().clamp_min(1.0)
            loss = (huber * mask * weight).sum() / denom
            if spec.aux > 0:
                _q, soc_hat, bd_hat = model.aux_at_mid(h, top, bot)
                soc_loss = (torch.nn.functional.huber_loss(soc_hat, soc, reduction="none") * mask * soc_m).sum() / (
                    mask * soc_m
                ).sum().clamp_min(1.0)
                bd_loss = (torch.nn.functional.huber_loss(bd_hat, bd, reduction="none") * mask * bd_m).sum() / (
                    mask * bd_m
                ).sum().clamp_min(1.0)
                loss = loss + spec.aux * (soc_loss / 10.0 + bd_loss)
            if spec.closure > 0:
                q_mid, soc_hat, bd_hat = model.aux_at_mid(h, top, bot)
                target_q = 0.01 * soc * bd
                clos_loss = ((q_mid - target_q) ** 2 * mask * closure).sum() / (mask * closure).sum().clamp_min(1.0)
                loss = loss + spec.closure * clos_loss
            if spec.smooth > 0:
                z = torch.linspace(0, 200, spec.smooth_nodes, device=device)
                h_exp = h[:, None, :].expand(-1, z.numel(), -1)
                q_s = model.q_at(h_exp.reshape(-1, h.shape[-1]), z.repeat(h.shape[0])).reshape(h.shape[0], -1)
                d2 = q_s[:, 2:] - 2 * q_s[:, 1:-1] + q_s[:, :-2]
                loss = loss + spec.smooth * (d2**2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        mae = _masked_mae(model, cache, val_ids, max_layers, device)
        if mae < best_mae - 1e-4:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= spec.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, max_layers, device


def _masked_mae(model, cache, ids, max_layers, device) -> float:
    if len(ids) == 0:
        return float("inf")
    model.eval()
    errs, weights = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            batch_ids = np.asarray(ids)[start : start + 64]
            x, top, bot, socd, _soc, _bd, mask, *_rest = _batch_from_cache(cache, batch_ids, max_layers, device)
            pred = model.integrate(model.encode(x), top, bot)
            errs.append((pred - socd).abs().detach().cpu().numpy())
            weights.append(mask.detach().cpu().numpy())
    err = np.concatenate(errs)
    mask = np.concatenate(weights)
    return float(err[mask > 0].mean())


def predict_layers(model, frame, x_by_profile, ids, max_layers, device) -> np.ndarray:
    cache, max_layers = build_profile_cache(frame, x_by_profile)
    model.eval()
    out = {}
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            batch_ids = np.asarray(list(ids))[start : start + 64]
            x, top, bot, socd, _soc, _bd, mask, *_rest = _batch_from_cache(cache, batch_ids, max_layers, device)
            pred = model.integrate(model.encode(x), top, bot).detach().cpu().numpy()
            layer_mask = mask.detach().cpu().numpy()
            for local, pid in enumerate(batch_ids):
                n = int(layer_mask[local].sum())
                out[str(pid)] = pred[local, :n]
    return out


def predict_stock(model, x_vector, depth_cm: float, device) -> float:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_vector, dtype=torch.float32, device=device).unsqueeze(0)
        h = model.encode(x)
        top = torch.zeros((1, 1), device=device)
        bot = torch.full((1, 1), float(depth_cm), device=device)
        return float(model.integrate(h, top, bot).cpu().item())


def ensemble_predict_layers(models, frame, x_by_profile, ids, max_layers, device):
    stack = []
    for model in models:
        pred = predict_layers(model, frame, x_by_profile, ids, max_layers, device)
        stack.append(pred)
    mean, std = {}, {}
    for pid in ids:
        key = str(pid)
        arr = np.stack([item[key] for item in stack], axis=0)
        mean[key] = arr.mean(axis=0)
        std[key] = arr.std(axis=0)
    return mean, std


def predict_stock_batch(model, x_matrix: np.ndarray, depth_cm: float, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(np.asarray(x_matrix), dtype=torch.float32, device=device)
        h = model.encode(x)
        top = torch.zeros((x.shape[0], 1), device=device)
        bot = torch.full((x.shape[0], 1), float(depth_cm), device=device)
        return model.integrate(h, top, bot).detach().cpu().numpy()[:, 0]


def ensemble_predict_stock(models, x_vector, depth_cm: float, device) -> tuple[float, float]:
    vals = [predict_stock(model, x_vector, depth_cm, device) for model in models]
    return float(np.mean(vals)), float(np.std(vals))
