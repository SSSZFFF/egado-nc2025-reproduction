"""
Physics-Residual Online Surrogate Digitizer v3 (Stage 2 + Stage 3 Unified)
==========================================================================
Upgrades over v2 (based on 2-mode B_strict ablation diagnosis):
  1. Neighborhood-Triggered Coordinate Tabu + Accept Cooldown:
     - Rejected (r, c) is locked out until an accepted flip occurs within its
       Chebyshev neighborhood (`tabu_unlock_radius`, default 3 => 7x7 window).
     - Eliminates repeatedly burning FDTD solves on dead outer-corner pixels
       like (5, 20) and stops ping-ponging like (16, 22).
  2. Expected Improvement (EI) Acquisition Function:
     - Replaces raw UCB (`mean + kappa * std`), which suffered from "dead-corner
       bias" (weak-field corners have mean ~ 0, while high-field core cells have
       negative mean but large upside tail).
     - EI above `ei_threshold` (default 1e-4, aligned with `tol`) naturally favors high-upside core
       pixels over dead corners that can only yield +1e-4.
  3. True Adjoint-Gradient Residual Support + Optional Periodic Refresh:
     - Properly tracks `_has_true_grad` even when `e2_map` is omitted.
     - Optional `refresh_physics_fn` re-computes exact adjoint `grad_eps` / `e2_field`
       every `refresh_every_flips` accepted flips (e.g. 4 FDTD solves in 2-mode).

Self-test:
    python physics_surrogate_digitizer.py --self-test
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# 1. Stage-1 / Discrete-Adjoint Physics Context
# =============================================================================

def downsample_to_blocks(arr_fine: np.ndarray, rows: int, cols: int, mode: str = "mean") -> np.ndarray:
    """Aggregate fine mesh (e.g. 20 nm TO or FDTD monitor) onto digital blocks (30x30 or 50x50)."""
    arr = np.asarray(arr_fine, dtype=np.float32)
    if arr.shape == (rows, cols):
        return arr.copy()

    h, w = arr.shape
    y_edges = np.linspace(0, h, rows + 1).astype(int)
    x_edges = np.linspace(0, w, cols + 1).astype(int)
    out = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        y0, y1 = y_edges[r], max(y_edges[r + 1], y_edges[r] + 1)
        for c in range(cols):
            x0, x1 = x_edges[c], max(x_edges[c + 1], x_edges[c] + 1)
            patch = arr[y0:y1, x0:x1]
            if mode == "sum":
                out[r, c] = float(np.sum(patch))
            elif mode == "max":
                out[r, c] = float(np.max(patch))
            else:
                out[r, c] = float(np.mean(patch))
    return out


def normalize_field(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale physics map to O(1) range for stable small-sample online learning."""
    arr = np.asarray(arr, dtype=np.float32)
    scale = float(np.percentile(np.abs(arr), 99)) + eps
    return np.clip(arr / scale, -3.0, 3.0).astype(np.float32)


@dataclass
class PhysicsContext:
    """
    Physical maps aligned to the digital (rows, cols) grid:
      - `eps_analog`: Continuous density in [0, 1] (or raw permittivity, auto-normalized).
      - `grad_map`:   Signed adjoint sensitivity dJ/deps (+ => adding Si increases J).
                      Shape (rows, cols) or (num_bands, rows, cols).
      - `e2_map`:     Forward field intensity |E|^2, shape (rows, cols) or (num_bands, rows, cols).
    """

    eps_analog: np.ndarray
    grad_map: Optional[np.ndarray] = None
    e2_map: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.eps_analog = np.asarray(self.eps_analog, dtype=np.float32)
        if self.eps_analog.max() > 1.5:
            lo, hi = float(self.eps_analog.min()), float(self.eps_analog.max())
            self.eps_analog = (self.eps_analog - lo) / max(hi - lo, 1e-8)
        self.eps_analog = np.clip(self.eps_analog, 0.0, 1.0)

        rows, cols = self.eps_analog.shape
        self._has_true_grad = self.grad_map is not None

        if self.grad_map is not None:
            g = np.asarray(self.grad_map, dtype=np.float32)
            if g.ndim == 2:
                g = g[None, :, :]
            self.grad_map = np.stack([normalize_field(g[b]) for b in range(g.shape[0])], axis=0)
        else:
            gy, gx = np.gradient(self.eps_analog)
            edge_mag = np.sqrt(gx * gx + gy * gy)
            self.grad_map = normalize_field(edge_mag)[None, :, :]

        if self.e2_map is not None:
            e2 = np.asarray(self.e2_map, dtype=np.float32)
            if e2.ndim == 2:
                e2 = e2[None, :, :]
            self.e2_map = np.stack([normalize_field(e2[b]) for b in range(e2.shape[0])], axis=0)
        else:
            self.e2_map = np.zeros((1, rows, cols), dtype=np.float32)

    @property
    def has_true_adjoint(self) -> bool:
        return bool(self._has_true_grad)

    def linear_prior(self, silicon_state: np.ndarray, r: int, c: int) -> float:
        """
        First-order physics prior for flipping cell (r, c):
          - With signed adjoint gradient: sum_b grad_b(r, c) * delta_si
          - Without adjoint gradient (fallback): analog density disagreement
        """
        cur = 1.0 if bool(silicon_state[r, c]) else 0.0
        delta_si = (1.0 - cur) - cur  # +1 for 0->1, -1 for 1->0
        if self.has_true_adjoint:
            return float(np.sum(self.grad_map[:, r, c]) * delta_si)
        analog_val = float(self.eps_analog[r, c])
        uncertainty = 1.0 - abs(2.0 * analog_val - 1.0)
        alignment_gain = (analog_val - 0.5) * delta_si
        return float(0.5 * uncertainty + alignment_gain)

    def extract_patch(
        self,
        silicon_state: np.ndarray,
        r: int,
        c: int,
        patch_size: int = 7,
    ) -> np.ndarray:
        pad = patch_size // 2
        cur_f = silicon_state.astype(np.float32)
        delta_si = 1.0 if not bool(silicon_state[r, c]) else -1.0

        def _crop(mat2d: np.ndarray, pad_val: float = 0.0) -> np.ndarray:
            padded = np.pad(mat2d, pad_width=pad, mode="constant", constant_values=pad_val)
            return padded[r : r + patch_size, c : c + patch_size]

        flip_plane = np.zeros((patch_size, patch_size), dtype=np.float32)
        flip_plane[pad, pad] = delta_si

        channels = [
            _crop(cur_f, pad_val=0.0),
            _crop(self.eps_analog, pad_val=0.0),
            flip_plane,
        ]
        for b in range(self.grad_map.shape[0]):
            channels.append(_crop(self.grad_map[b], pad_val=0.0))
        for b in range(self.e2_map.shape[0]):
            channels.append(_crop(self.e2_map[b], pad_val=0.0))

        return np.stack(channels, axis=0).astype(np.float32)

    def extract_physics_features(self, silicon_state: np.ndarray, r: int, c: int) -> np.ndarray:
        """6-dim rotation/translation-invariant physics features (avoids d >> n overfitting in Ridge)."""
        rows, cols = silicon_state.shape
        cur = 1.0 if bool(silicon_state[r, c]) else 0.0
        delta_si = (1.0 - cur) - cur  # +1 (0->1) or -1 (1->0)
        lin_prior = self.linear_prior(silicon_state, r, c)
        center_e2 = float(np.mean(self.e2_map[:, r, c]))

        r0, r1 = max(0, r - 1), min(rows, r + 2)
        c0, c1 = max(0, c - 1), min(cols, c + 2)
        nbr_grad = float(np.mean(self.grad_map[:, r0:r1, c0:c1]) * delta_si)
        nbr_si_frac = float(np.mean(silicon_state[r0:r1, c0:c1]))
        abs_prior = abs(lin_prior)

        return np.array(
            [lin_prior, center_e2, nbr_grad, nbr_si_frac, delta_si, abs_prior],
            dtype=np.float32,
        )


# =============================================================================
# 2. Acquisition Function (Gaussian Expected Improvement, pure NumPy)
# =============================================================================

def expected_improvement(mean: np.ndarray, std: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    """
    Compute Gaussian Expected Improvement E[max(0, dJ - threshold)]:
      - Suppresses weak-field dead corners where mean ~ 0 and std ~ 0.
      - Rewards high-upside core pixels with positive mean OR large upside tail.
    """
    mu = np.asarray(mean, dtype=np.float64)
    sig = np.maximum(np.asarray(std, dtype=np.float64), 1e-9)
    z = (mu - threshold) / sig
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    inv_sqrt2pi = 1.0 / math.sqrt(2.0 * math.pi)
    cdf = np.array([0.5 * (1.0 + math.erf(v * inv_sqrt2)) for v in z], dtype=np.float64)
    pdf = inv_sqrt2pi * np.exp(-0.5 * z * z)
    ei = (mu - threshold) * cdf + sig * pdf
    return np.maximum(ei, 0.0).astype(np.float32)


# =============================================================================
# 3. Lightweight Physics-Residual Surrogate (PyTorch CNN + NumPy Ridge Fallback)
# =============================================================================

if HAS_TORCH:

    class _LocalPatchNet(nn.Module):
        def __init__(self, in_channels: int, hidden: int = 24):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(hidden * 2 + 1, 32)
            self.fc2 = nn.Linear(32, 1)
            self.prior_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        def forward(self, x: torch.Tensor, linear_prior: torch.Tensor) -> torch.Tensor:
            h = F.silu(self.conv1(x))
            h = F.silu(self.conv2(h))
            h = F.silu(self.conv3(h))
            center = h[:, :, h.shape[2] // 2, h.shape[3] // 2]
            avg = h.mean(dim=(2, 3))
            feat = torch.cat([center, avg, linear_prior.unsqueeze(-1)], dim=-1)
            residual = self.fc2(F.silu(self.fc1(feat))).squeeze(-1)
            return self.prior_scale * linear_prior + residual


class OnlinePhysicsSurrogate:
    """
    Ensemble surrogate predicting (mean_dJ, std_dJ) for candidate pixel flips.
    Uses PyTorch CNN ensemble when available, otherwise bootstrapped Ridge ensemble
    with explicit prior-weight emphasis for small-sample stability.
    """

    def __init__(
        self,
        in_channels: int,
        ensemble_size: int = 5,
        lr: float = 3e-3,
        weight_decay: float = 1e-3,
        ridge_lambda: float = 2.0,
        min_samples: int = 8,
        pure_adjoint: bool = False,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.ensemble_size = ensemble_size
        self.ridge_lambda = ridge_lambda
        self.min_samples = min_samples
        self.pure_adjoint = pure_adjoint
        self.rng = np.random.default_rng(seed)
        self.patches: list[np.ndarray] = []
        self.priors: list[float] = []
        self.targets: list[float] = []
        self.use_torch = HAS_TORCH

        if self.use_torch:
            torch.manual_seed(seed)
            self.device = torch.device(
                device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self.models = [_LocalPatchNet(in_channels).to(self.device) for _ in range(ensemble_size)]
            self.optimizers = [
                torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay) for m in self.models
            ]
        else:
            self.ridge_weights: list[Optional[np.ndarray]] = [None for _ in range(ensemble_size)]

    def __len__(self) -> int:
        return len(self.targets)

    def add_observation(self, patch: np.ndarray, linear_prior: float, true_delta_j: float) -> None:
        self.patches.append(patch.astype(np.float32))
        self.priors.append(float(linear_prior))
        self.targets.append(float(true_delta_j))

    def reset_replay(self, keep_last: int = 8) -> None:
        """Drop stale observations after a fresh adjoint refresh.

        Keeps only the most recent `keep_last` samples so the new exact linear
        prior leads the next acquisition round instead of old-structure labels.
        """
        if keep_last <= 0:
            self.patches.clear()
            self.priors.clear()
            self.targets.clear()
            return
        self.patches = self.patches[-keep_last:]
        self.priors = self.priors[-keep_last:]
        self.targets = self.targets[-keep_last:]

    def fit_online(self, steps: int = 20, batch_size: int = 32) -> float:
        n = len(self.targets)
        if self.pure_adjoint or n < self.min_samples:
            return 0.0

        X = np.stack(self.patches, axis=0)
        P = np.asarray(self.priors, dtype=np.float32)
        y = np.asarray(self.targets, dtype=np.float32)

        if self.use_torch:
            X_t = torch.from_numpy(X).to(self.device)
            P_t = torch.from_numpy(P).to(self.device)
            y_t = torch.from_numpy(y).to(self.device)
            total_loss = 0.0
            for model, opt in zip(self.models, self.optimizers):
                model.train()
                for _ in range(steps):
                    bs = min(batch_size, n)
                    idx = self.rng.integers(0, n, size=bs)
                    if n > 8:
                        idx[: max(1, bs // 4)] = self.rng.integers(max(0, n - 8), n, size=max(1, bs // 4))
                    idx_t = torch.from_numpy(idx).long().to(self.device)
                    opt.zero_grad()
                    pred = model(X_t[idx_t], P_t[idx_t])
                    loss = F.smooth_l1_loss(pred, y_t[idx_t], beta=0.005)
                    loss.backward()
                    opt.step()
                    total_loss += float(loss.item())
            return total_loss / (self.ensemble_size * steps)

        # Pure NumPy Ridge ensemble (heavier regularization on raw patch, light reg on linear_prior)
        feats = np.concatenate([X.reshape(n, -1), P[:, None], np.ones((n, 1), dtype=np.float32)], axis=1)
        d = feats.shape[1]
        reg = self.ridge_lambda * np.eye(d, dtype=np.float32)
        reg[0, 0] = 0.05   # 6D feature lin_prior
        reg[5, 5] = 0.05   # 6D feature abs_prior (strong nonlinear amplitude)
        reg[-2, -2] = 0.05 # independent prior column
        reg[-1, -1] = 0.0  # bias
        for m_idx in range(self.ensemble_size):
            idx = self.rng.integers(0, n, size=n)
            Xm, ym = feats[idx], y[idx]
            self.ridge_weights[m_idx] = np.linalg.solve(Xm.T @ Xm + reg, Xm.T @ ym)
        return 0.0

    def predict(self, patches: Sequence[np.ndarray], priors: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        m = len(patches)
        P = np.asarray(priors, dtype=np.float32)
        if self.pure_adjoint or len(self.targets) < self.min_samples:
            return 0.01 * np.abs(P), np.full(m, 1e-4, dtype=np.float32)

        X = np.stack(patches, axis=0)
        if self.use_torch:
            X_t = torch.from_numpy(X).to(self.device)
            P_t = torch.from_numpy(P).to(self.device)
            preds = []
            with torch.no_grad():
                for model in self.models:
                    model.eval()
                    preds.append(model(X_t, P_t).cpu().numpy())
            stack = np.stack(preds, axis=0)
            return stack.mean(axis=0), stack.std(axis=0)

        feats = np.concatenate([X.reshape(m, -1), P[:, None], np.ones((m, 1), dtype=np.float32)], axis=1)
        preds = [feats @ w for w in self.ridge_weights if w is not None]
        stack = np.stack(preds, axis=0)
        return stack.mean(axis=0), stack.std(axis=0)


# =============================================================================
# 4. Unified Stage 2 + Stage 3 Optimizer (Tabu + EI + Optional Adjoint Refresh)
# =============================================================================

class SurrogateGuidedDigitizer:
    def __init__(
        self,
        initial_state: np.ndarray,
        physics_ctx: PhysicsContext,
        check_drc_fn: Callable[[np.ndarray], bool],
        eval_fn: Callable[[np.ndarray], float],
        out_dir: Path,
        candidate_mask: Optional[np.ndarray] = None,
        verify_drc_fn: Optional[Callable[[np.ndarray], bool]] = None,
        on_accept_fn: Optional[Callable[[np.ndarray, float, int], None]] = None,
        pure_adjoint: bool = False,
        compact_features: Optional[bool] = None,
        refresh_physics_fn: Optional[Callable[[np.ndarray], PhysicsContext]] = None,
        refresh_every_flips: int = 5,
        refresh_no_improve_rounds: int = 4,
        patch_size: int = 7,
        top_k: int = 2,
        mag_top_n: int = 36,
        acquisition: str = "ei",
        ei_threshold: float = 1e-4,
        ucb_kappa: float = 0.5,
        tabu_unlock_radius: int = 3,
        accept_cooldown_rounds: int = 5,
        tol: float = 1e-4,
        min_spatial_sep: int = 2,
        max_rounds: int = 120,
        max_fdtd_evals: int = 150,
        patience_rounds: int = 12,
        seed: int = 42,
    ):
        self.state = initial_state.astype(bool).copy()
        self.rows, self.cols = self.state.shape
        self.ctx = physics_ctx
        self.check_drc_fn = check_drc_fn
        self.verify_drc_fn = verify_drc_fn  # Optional Tier-2 full 248-rule PDK DRC
        self.eval_fn = eval_fn
        self.on_accept_fn = on_accept_fn
        self.pure_adjoint = pure_adjoint
        # No PyTorch => use the 6D compact physics features. With PyTorch
        # the CNN path can continue to use the 7x7 multichannel patch.
        self.compact_features = (not HAS_TORCH) if compact_features is None else compact_features
        self.refresh_physics_fn = refresh_physics_fn
        self.refresh_every_flips = refresh_every_flips
        self.refresh_no_improve_rounds = refresh_no_improve_rounds
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.mask = (
            np.ones((self.rows, self.cols), dtype=bool)
            if candidate_mask is None
            else candidate_mask.astype(bool).copy()
        )
        self.patch_size = patch_size
        self.top_k = top_k
        self.mag_top_n = mag_top_n
        self.acquisition = acquisition.lower()
        self.ei_threshold = ei_threshold
        self.ucb_kappa = ucb_kappa
        self.tabu_unlock_radius = tabu_unlock_radius
        self.accept_cooldown_rounds = accept_cooldown_rounds
        self.tol = tol
        self.min_spatial_sep = min_spatial_sep
        self.max_rounds = max_rounds
        self.max_fdtd_evals = max_fdtd_evals
        self.patience_rounds = patience_rounds

        sample_patch = self.ctx.extract_patch(self.state, 0, 0, patch_size=self.patch_size)
        self.surrogate = OnlinePhysicsSurrogate(
            in_channels=6 if self.compact_features else sample_patch.shape[0],
            pure_adjoint=pure_adjoint,
            seed=seed,
        )

        self.eval_cache: dict[bytes, float] = {}
        self.tabu_coords: set[tuple[int, int]] = set()
        self.accept_cooldown: dict[tuple[int, int], int] = {}
        self.fdtd_evals = 0
        self.total_flips = 0
        self.tabu_resets = 0
        self.adjoint_refreshes = 0
        self.refresh_errors = 0
        self.last_refresh_flip_count = 0

    def _eval_cached(self, state: np.ndarray) -> tuple[float, bool]:
        key = np.packbits(state.astype(np.uint8)).tobytes()
        if key in self.eval_cache:
            return self.eval_cache[key], True
        j_val = float(self.eval_fn(state))
        self.eval_cache[key] = j_val
        self.fdtd_evals += 1
        return j_val, False

    def _unlock_neighborhood(self, center_r: int, center_c: int) -> None:
        rad = self.tabu_unlock_radius
        to_unlock = [
            (r, c)
            for (r, c) in self.tabu_coords
            if max(abs(r - center_r), abs(c - center_c)) <= rad
        ]
        for coord in to_unlock:
            self.tabu_coords.discard(coord)

    def _score_candidates(self, mean_dj: np.ndarray, std_dj: np.ndarray, priors: Optional[Sequence[float]] = None) -> np.ndarray:
        prior_arr = np.asarray(priors, dtype=np.float32) if priors is not None else np.zeros(len(mean_dj), dtype=np.float32)
        if self.pure_adjoint:
            return np.abs(prior_arr)
        if self.acquisition == "ei":
            scores = expected_improvement(mean_dj, std_dj, threshold=self.ei_threshold)
            # Tie-breaker: when EI collapses to 0, still prefer high-|g| core
            # pixels, independent of the sign of the first-order prior.
            if priors is not None:
                scores = scores + 1e-6 * np.abs(prior_arr)
            return scores
        return mean_dj + self.ucb_kappa * std_dj

    def _refresh_physics(self) -> bool:
        if self.refresh_physics_fn is None:
            return False
        # Never refresh the same accepted state twice. This prevents the
        # no-improve trigger from burning solves on an unchanged layout.
        if self.total_flips <= self.last_refresh_flip_count:
            return False
        try:
            new_ctx = self.refresh_physics_fn(self.state)
        except Exception as exc:
            self.refresh_errors += 1
            print(f"[Surrogate] physics refresh failed (ignored): {exc}", flush=True)
            return False
        if new_ctx is not None:
            self.ctx = new_ctx
        self.adjoint_refreshes += 1
        self.last_refresh_flip_count = self.total_flips
        self.surrogate.reset_replay(keep_last=8)
        self.tabu_coords.clear()
        return True

    def run(self) -> dict:
        t0 = time.time()
        current_j, _ = self._eval_cached(self.state)
        best_j = current_j
        no_improve_rounds = 0

        hist_path = self.out_dir / "surrogate_history.csv"
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "round",
                    "row",
                    "col",
                    "flipped_to",
                    "pred_dJ",
                    "pred_std",
                    "acq_score",
                    "true_dJ",
                    "J_trial",
                    "J_current",
                    "accepted",
                    "fdtd_evals",
                    "hit_rate",
                    "tabu_size",
                    "wall_time_s",
                ]
            )

        all_mask_coords = [tuple(int(v) for v in x) for x in np.argwhere(self.mask)]

        for rnd in range(1, self.max_rounds + 1):
            if self.fdtd_evals >= self.max_fdtd_evals:
                break

            # 1. Fast Tier-1 DRC + Tabu/Cooldown filtering
            legal_coords: list[tuple[int, int]] = []
            legal_patches: list[np.ndarray] = []
            legal_priors: list[float] = []

            for r, c in all_mask_coords:
                if (r, c) in self.tabu_coords:
                    continue
                if self.accept_cooldown.get((r, c), 0) >= rnd:
                    continue
                trial = self.state.copy()
                trial[r, c] = not trial[r, c]
                key = np.packbits(trial.astype(np.uint8)).tobytes()
                if key in self.eval_cache:
                    continue
                if not self.check_drc_fn(trial):
                    continue
                legal_coords.append((r, c))
                feat = (
                    self.ctx.extract_physics_features(self.state, r, c)
                    if self.compact_features
                    else self.ctx.extract_patch(self.state, r, c, self.patch_size)
                )
                legal_patches.append(feat)
                legal_priors.append(self.ctx.linear_prior(self.state, r, c))

            # Magnitude-first core-pool filter: do not let signed first-order
            # priors bury strong nonlinear pixels such as (23,21) at rank 99.
            if self.mag_top_n > 0 and len(legal_coords) > self.mag_top_n:
                mag_order = np.argsort(-np.abs(np.asarray(legal_priors, dtype=np.float32)))[: self.mag_top_n]
                legal_coords = [legal_coords[int(i)] for i in mag_order]
                legal_patches = [legal_patches[int(i)] for i in mag_order]
                legal_priors = [legal_priors[int(i)] for i in mag_order]

            if not legal_coords:
                if self.tabu_coords and self.tabu_resets < 1:
                    self.tabu_coords.clear()
                    self.tabu_resets += 1
                    continue
                break

            # 2. Batched surrogate scoring (milliseconds)
            mean_dj, std_dj = self.surrogate.predict(legal_patches, legal_priors)
            acq_scores = self._score_candidates(mean_dj, std_dj, legal_priors)
            order = np.argsort(-acq_scores)

            # 3. Pick Top-K with spatial diversity + Tier-2 PDK DRC verification (skip & backfill)
            k_budget = min(self.top_k, self.max_fdtd_evals - self.fdtd_evals, len(legal_coords))
            chosen_indices: list[int] = []
            for pass_sep in (self.min_spatial_sep, 0):
                if len(chosen_indices) >= k_budget:
                    break
                for raw_idx in order:
                    idx = int(raw_idx)
                    if idx in chosen_indices:
                        continue
                    r, c = legal_coords[idx]
                    if pass_sep > 0 and any(
                        max(abs(r - legal_coords[j][0]), abs(c - legal_coords[j][1])) < pass_sep
                        for j in chosen_indices
                    ):
                        continue
                    trial = self.state.copy()
                    trial[r, c] = not trial[r, c]
                    if self.verify_drc_fn is not None and not self.verify_drc_fn(trial):
                        self.tabu_coords.add((r, c))
                        continue
                    chosen_indices.append(idx)
                    if len(chosen_indices) >= k_budget:
                        break

            if not chosen_indices:
                break

            # 4. Evaluate Top-K candidates with real FDTD
            round_results = []
            for idx in chosen_indices:
                r, c = legal_coords[idx]
                trial = self.state.copy()
                trial[r, c] = not trial[r, c]
                j_trial, _ = self._eval_cached(trial)
                true_dj = j_trial - current_j
                self.surrogate.add_observation(legal_patches[idx], legal_priors[idx], true_dj)
                round_results.append(
                    (idx, r, c, trial, j_trial, true_dj, float(mean_dj[idx]), float(std_dj[idx]), float(acq_scores[idx]))
                )

            # 5. Online fine-tuning on replay buffer
            self.surrogate.fit_online(steps=20, batch_size=32)

            # 6. Accept best flip & update Tabu/Cooldown
            round_results.sort(key=lambda item: item[4], reverse=True)
            best_item = round_results[0]
            accepted_any = best_item[4] > current_j + self.tol

            if accepted_any:
                _, br, bc, best_trial, best_trial_j, _, _, _, _ = best_item
                self.state = best_trial
                current_j = best_trial_j
                best_j = max(best_j, current_j)
                self.total_flips += 1
                no_improve_rounds = 0

                self._unlock_neighborhood(br, bc)
                self.accept_cooldown[(br, bc)] = rnd + self.accept_cooldown_rounds
                for item in round_results[1:]:
                    _, rr, rc = item[0], item[1], item[2]
                    if max(abs(rr - br), abs(rc - bc)) > self.tabu_unlock_radius:
                        self.tabu_coords.add((rr, rc))

                np.savez(
                    self.out_dir / f"state_round{rnd}_flip{self.total_flips}.npz",
                    silicon_state=self.state,
                    candidate_mask=self.mask,
                    J=current_j,
                )
                if self.on_accept_fn is not None:
                    self.on_accept_fn(self.state, current_j, self.total_flips)

                if (
                    self.refresh_physics_fn is not None
                    and self.refresh_every_flips > 0
                    and self.total_flips % self.refresh_every_flips == 0
                ):
                    self._refresh_physics()
            else:
                no_improve_rounds += 1
                for item in round_results:
                    self.tabu_coords.add((item[1], item[2]))
                if (
                    self.refresh_physics_fn is not None
                    and self.refresh_no_improve_rounds > 0
                    and no_improve_rounds >= self.refresh_no_improve_rounds
                ):
                    if self._refresh_physics():
                        no_improve_rounds = 0

            hit_rate = self.total_flips / max(1, self.fdtd_evals - 1)
            wall_s = time.time() - t0
            with open(hist_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for idx, r, c, trial, j_trial, true_dj, p_mean, p_std, acq in round_results:
                    is_acc = accepted_any and (r == best_item[1] and c == best_item[2])
                    writer.writerow(
                        [
                            rnd,
                            r,
                            c,
                            int(trial[r, c]),
                            f"{p_mean:.6f}",
                            f"{p_std:.6f}",
                            f"{acq:.6f}",
                            f"{true_dj:.6f}",
                            f"{j_trial:.6f}",
                            f"{current_j:.6f}",
                            int(is_acc),
                            self.fdtd_evals,
                            f"{hit_rate:.4f}",
                            len(self.tabu_coords),
                            f"{wall_s:.2f}",
                        ]
                    )

            if no_improve_rounds >= self.patience_rounds:
                break

        summary = {
            "final_J": best_j,
            "total_flips": self.total_flips,
            "fdtd_evals": self.fdtd_evals,
            "hit_rate": self.total_flips / max(1, self.fdtd_evals - 1),
            "acquisition": self.acquisition,
            "pure_adjoint": self.pure_adjoint,
            "compact_features": self.compact_features,
            "has_true_adjoint": self.ctx.has_true_adjoint,
            "adjoint_refreshes": self.adjoint_refreshes,
            "refresh_errors": self.refresh_errors,
            "tabu_resets": self.tabu_resets,
            "backend": "pytorch_cnn_ensemble" if self.surrogate.use_torch else "numpy_ridge_ensemble",
            "wall_time_s": round(time.time() - t0, 2),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez(
            self.out_dir / "final_state.npz",
            silicon_state=self.state,
            candidate_mask=self.mask,
            J=best_j,
        )
        return summary


class PasswiseMagnitudeDigitizer:
    """
    v5: Pass-wise Adjoint-Magnitude Guided Digital Search (MG-DS).
    - Pass 1 uses initial `physics_ctx` (0 extra solves at start).
    - Within each pass, visits the Top-N `|linear_prior|` pixels at most ONCE (no loops).
    - Between passes, refreshes exact adjoint ONLY if >= 1 flip was accepted in the
      previous pass, then re-ranks Top-N and re-enables coupled pixels (e.g. (23, 21)).
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        physics_ctx: PhysicsContext,
        check_drc_fn: Callable[[np.ndarray], bool],
        eval_fn: Callable[[np.ndarray], float],
        out_dir: Path,
        candidate_mask: Optional[np.ndarray] = None,
        verify_drc_fn: Optional[Callable[[np.ndarray], bool]] = None,
        on_accept_fn: Optional[Callable[[np.ndarray, float, int], None]] = None,
        refresh_physics_fn: Optional[Callable[[np.ndarray], PhysicsContext]] = None,
        pass_top_n: Sequence[int] = (36, 36, 48),
        tol: float = 1e-4,
        max_fdtd_evals: int = 120,
    ):
        self.state = initial_state.astype(bool).copy()
        self.rows, self.cols = self.state.shape
        self.ctx = physics_ctx
        self.check_drc_fn = check_drc_fn
        self.verify_drc_fn = verify_drc_fn
        self.eval_fn = eval_fn
        self.on_accept_fn = on_accept_fn
        self.refresh_physics_fn = refresh_physics_fn
        self.pass_top_n = list(pass_top_n)
        self.tol = tol
        self.max_fdtd_evals = max_fdtd_evals
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.mask = (
            np.ones((self.rows, self.cols), dtype=bool)
            if candidate_mask is None
            else candidate_mask.astype(bool).copy()
        )
        self.eval_cache: dict[bytes, float] = {}
        self.fdtd_evals = 0
        self.total_flips = 0
        self.adjoint_refreshes = 0
        self.refresh_errors = 0

    def _eval_cached(self, state: np.ndarray) -> tuple[float, bool]:
        key = np.packbits(state.astype(np.uint8)).tobytes()
        if key in self.eval_cache:
            return self.eval_cache[key], True
        j_val = float(self.eval_fn(state))
        self.eval_cache[key] = j_val
        self.fdtd_evals += 1
        return j_val, False

    def run(self) -> dict:
        t0 = time.time()
        current_j, _ = self._eval_cached(self.state)
        best_j = current_j

        hist_path = self.out_dir / "surrogate_history.csv"
        with open(hist_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "pass_idx",
                    "queue_rank",
                    "row",
                    "col",
                    "flipped_to",
                    "signed_prior",
                    "abs_prior",
                    "true_dJ",
                    "J_trial",
                    "J_current",
                    "accepted",
                    "fdtd_evals",
                    "hit_rate",
                    "wall_time_s",
                ]
            )

        all_coords = [tuple(int(v) for v in x) for x in np.argwhere(self.mask)]

        for p_idx, top_n in enumerate(self.pass_top_n, start=1):
            if self.fdtd_evals >= self.max_fdtd_evals:
                break

            # Build ordered queue by |linear_prior| descending (independent of sign)
            scored = []
            for r, c in all_coords:
                sp = self.ctx.linear_prior(self.state, r, c)
                scored.append((abs(sp), sp, r, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            pass_queue = scored[: min(top_n, len(scored))]

            accepted_in_pass = 0
            for q_rank, (abs_p, signed_p, r, c) in enumerate(pass_queue, start=1):
                if self.fdtd_evals >= self.max_fdtd_evals:
                    break

                trial = self.state.copy()
                trial[r, c] = not trial[r, c]
                key = np.packbits(trial.astype(np.uint8)).tobytes()
                if key in self.eval_cache:
                    continue
                if not self.check_drc_fn(trial):
                    continue
                if self.verify_drc_fn is not None and not self.verify_drc_fn(trial):
                    continue

                j_trial, _ = self._eval_cached(trial)
                true_dj = j_trial - current_j
                accepted = j_trial > current_j + self.tol

                if accepted:
                    self.state = trial
                    current_j = j_trial
                    best_j = max(best_j, current_j)
                    self.total_flips += 1
                    accepted_in_pass += 1
                    np.savez(
                        self.out_dir / f"state_pass{p_idx}_flip{self.total_flips}.npz",
                        silicon_state=self.state,
                        candidate_mask=self.mask,
                        J=current_j,
                    )
                    if self.on_accept_fn is not None:
                        self.on_accept_fn(self.state, current_j, self.total_flips)

                hit_rate = self.total_flips / max(1, self.fdtd_evals - 1)
                with open(hist_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(
                        [
                            p_idx,
                            q_rank,
                            r,
                            c,
                            int(trial[r, c]),
                            f"{signed_p:.6f}",
                            f"{abs_p:.6f}",
                            f"{true_dj:.6f}",
                            f"{j_trial:.6f}",
                            f"{current_j:.6f}",
                            int(accepted),
                            self.fdtd_evals,
                            f"{hit_rate:.4f}",
                            f"{time.time() - t0:.2f}",
                        ]
                    )

            # Stop early if a full pass yielded zero accepted flips
            if accepted_in_pass == 0:
                break

            # Refresh exact adjoint ONCE between passes (only when structure changed & more passes remain)
            if (
                p_idx < len(self.pass_top_n)
                and self.refresh_physics_fn is not None
                and self.fdtd_evals < self.max_fdtd_evals
            ):
                try:
                    new_ctx = self.refresh_physics_fn(self.state)
                except Exception as exc:
                    self.refresh_errors += 1
                    print(f"[Passwise] physics refresh failed (ignored): {exc}", flush=True)
                    new_ctx = None
                if new_ctx is not None:
                    self.ctx = new_ctx
                    self.adjoint_refreshes += 1

        summary = {
            "strategy": "passwise_magnitude",
            "pass_top_n": self.pass_top_n,
            "final_J": best_j,
            "total_flips": self.total_flips,
            "fdtd_evals": self.fdtd_evals,
            "hit_rate": self.total_flips / max(1, self.fdtd_evals - 1),
            "adjoint_refreshes": self.adjoint_refreshes,
            "refresh_errors": self.refresh_errors,
            "has_true_adjoint": self.ctx.has_true_adjoint,
            "wall_time_s": round(time.time() - t0, 2),
        }
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        np.savez(
            self.out_dir / "final_state.npz",
            silicon_state=self.state,
            candidate_mask=self.mask,
            J=best_j,
        )
        return summary


def _run_self_test() -> None:
    import tempfile

    rng = np.random.default_rng(7)
    rows, cols = 30, 30
    target_si = (rng.random((rows, cols)) > 0.5).astype(bool)
    eps_analog = np.clip(target_si.astype(np.float32) * 0.75 + 0.12 + rng.normal(0, 0.25, (rows, cols)), 0, 1)
    init_state = (eps_analog > 0.5).astype(bool)

    weights = np.full((rows, cols), 0.0002, dtype=np.float32)
    weights[8:22, 6:24] = 0.012
    e2_map = np.zeros((rows, cols), dtype=np.float32)
    e2_map[8:22, 6:24] = 1.0

    def synthetic_fom(state: np.ndarray) -> float:
        s = state.astype(np.float32)
        t = target_si.astype(np.float32)
        return 0.95 - float(np.sum(weights * np.abs(s - t)))

    grad_map = weights * (2.0 * target_si.astype(np.float32) - 1.0)

    def dummy_drc(_trial: np.ndarray) -> bool:
        return True

    for label, ctx in [
        ("FastStart (no grad) + EI + Tabu", PhysicsContext(eps_analog=eps_analog)),
        ("TrueAdjoint + EI + Tabu", PhysicsContext(eps_analog=eps_analog, grad_map=grad_map, e2_map=e2_map)),
    ]:
        with tempfile.TemporaryDirectory() as tmp:
            opt = SurrogateGuidedDigitizer(
                initial_state=init_state,
                physics_ctx=ctx,
                check_drc_fn=dummy_drc,
                eval_fn=synthetic_fom,
                out_dir=Path(tmp),
                top_k=2,
                mag_top_n=0,
                acquisition="ei",
                ei_threshold=5e-4,
                max_rounds=60,
                max_fdtd_evals=80,
                patience_rounds=10,
            )
            res = opt.run()
        print(
            f"[{label}] init_J={synthetic_fom(init_state):.4f} -> final_J={res['final_J']:.4f} | "
            f"flips={res['total_flips']} | evals={res['fdtd_evals']} | "
            f"hit_rate={res['hit_rate']*100:.1f}% | backend={res['backend']}"
        )


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        _run_self_test()
    else:
        print("Import SurrogateGuidedDigitizer & PhysicsContext, or run with --self-test.")
