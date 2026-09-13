"""
strategy_engine.simulator.regime_sde
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-asset Merton Jump-Diffusion Regime-Switching Stochastic Differential Equation (SDE)
simulator with Cholesky cross-asset correlation, log-Euler discretization, and OHLCV
candlestick synthesis.

SDE Model:
    dS_i / S_i = (mu_i - lambda_i * kappa_i) dt + sigma_i dW_i + (e^{J_i} - 1) dN_i
    where kappa_i = exp(mu_{J,i} + 0.5 * sigma_{J,i}^2) - 1 is the jump compensator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from strategy_engine.core.models import Bar


def make_positive_semidefinite(corr: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite correlation matrix."""
    with np.errstate(all="ignore"):
        corr_clean = np.nan_to_num(corr, nan=0.0, posinf=1.0, neginf=-1.0)
        sym = 0.5 * (corr_clean + corr_clean.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals_clipped = np.clip(eigvals, eps, None)
        psd = eigvecs @ (np.diag(eigvals_clipped) @ eigvecs.T)
        d = np.clip(np.diag(psd), eps, None)
        inv_sqrt_diag = 1.0 / np.sqrt(d)
        clean_corr = psd * np.outer(inv_sqrt_diag, inv_sqrt_diag)
        np.fill_diagonal(clean_corr, 1.0)
        return clean_corr


@dataclass(frozen=True)
class AssetJumpDiffusionParams:
    """Parameters for a single asset under Merton jump-diffusion."""
    symbol: str
    s0: float
    mu: float
    sigma: float
    lambda_j: float = 0.0
    mu_j: float = 0.0
    sigma_j: float = 0.0
    base_volume: int = 10_000_000


@dataclass(frozen=True)
class RegimeParameters:
    """Configuration for a market regime across multiple assets."""
    name: str
    assets: Dict[str, AssetJumpDiffusionParams]
    correlation_matrix: np.ndarray


@dataclass(frozen=True)
class PiecewiseRegimeSchedule:
    """Chronological regime phases (e.g. crash followed by recovery)."""
    phases: List[Tuple[int, RegimeParameters]]


class MertonJumpDiffusionSimulator:
    """High-performance, deterministic vector simulator for correlated multi-asset
    Merton Jump-Diffusion Regime-Switching SDEs.
    """

    def __init__(self, seed: Optional[int] = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset the internal pseudo-random number generator."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def compute_jump_compensator(mu_j: float, sigma_j: float) -> float:
        """Compute kappa = E[exp(J) - 1] = exp(mu_j + 0.5 * sigma_j^2) - 1."""
        return float(np.exp(mu_j + 0.5 * (sigma_j ** 2)) - 1.0)

    def simulate_multivariate_paths(
        self,
        symbols: List[str],
        initial_prices: Dict[str, float],
        drifts: Dict[str, float],
        volatilities: Dict[str, float],
        correlation_matrix: np.ndarray,
        jump_lambda: Union[float, Dict[str, float]] = 0.0,
        jump_mean: Union[float, Dict[str, float]] = 0.0,
        jump_vol: Union[float, Dict[str, float]] = 0.0,
        n_days: int = 252,
        dt: float = 1.0 / 252.0,
    ) -> Dict[str, np.ndarray]:
        """Simulate multivariate correlated Merton jump-diffusion price paths using exact log-Euler."""
        n_assets = len(symbols)
        clean_corr = make_positive_semidefinite(correlation_matrix)
        L = np.linalg.cholesky(clean_corr)

        lam_map = jump_lambda if isinstance(jump_lambda, dict) else {s: jump_lambda for s in symbols}
        mu_j_map = jump_mean if isinstance(jump_mean, dict) else {s: jump_mean for s in symbols}
        sig_j_map = jump_vol if isinstance(jump_vol, dict) else {s: jump_vol for s in symbols}

        kappa = {
            s: (self.compute_jump_compensator(mu_j_map[s], sig_j_map[s]) if lam_map[s] > 0 else 0.0)
            for s in symbols
        }
        drift_dt = {
            s: (drifts[s] - lam_map[s] * kappa[s] - 0.5 * (volatilities[s] ** 2)) * dt
            for s in symbols
        }

        paths = {s: np.zeros(n_days + 1, dtype=np.float64) for s in symbols}
        for s in symbols:
            paths[s][0] = initial_prices[s]

        sqrt_dt = math.sqrt(dt)

        for t in range(n_days):
            z = self.rng.standard_normal(n_assets)
            dw = np.dot(L, z) * sqrt_dt

            for i, s in enumerate(symbols):
                jumps = 0.0
                if lam_map[s] > 0:
                    dn = self.rng.poisson(lam_map[s] * dt)
                    if dn > 0:
                        j_draws = self.rng.normal(mu_j_map[s], sig_j_map[s], size=dn)
                        jumps = float(np.sum(j_draws))

                ret = drift_dt[s] + volatilities[s] * dw[i] + jumps
                paths[s][t + 1] = paths[s][t] * math.exp(ret)

        return paths

    def paths_to_bars(
        self,
        paths: Dict[str, np.ndarray],
        symbols: List[str],
        start_date: datetime,
        base_volumes: Optional[Dict[str, int]] = None,
    ) -> Dict[str, List[Bar]]:
        """Convert continuous simulated paths of length (n_days + 1) into exactly n_days Bar objects."""
        result: Dict[str, List[Bar]] = {s: [] for s in symbols}
        n_days = len(next(iter(paths.values()))) - 1

        if base_volumes is None:
            base_volumes = {
                "SPY": 65_000_000,
                "QQQ": 45_000_000,
                "AAPL": 50_000_000,
                "MSFT": 25_000_000,
                "NVDA": 40_000_000,
                "TLT": 20_000_000,
                "SHV": 5_000_000,
                "BIL": 4_000_000,
                "GLD": 8_000_000,
                "XLK": 12_000_000,
                "XLE": 22_000_000,
                "XLF": 35_000_000,
            }

        cur_date = start_date
        for t in range(n_days):
            while cur_date.weekday() >= 5:
                cur_date += timedelta(days=1)

            bar_ts = cur_date.replace(hour=16, minute=0, second=0, microsecond=0)

            for s in symbols:
                o = float(paths[s][t])
                c = float(paths[s][t + 1])
                min_oc = min(o, c)
                max_oc = max(o, c)

                if s in ("SHV", "BIL", "CASH"):
                    intra_spread = min(0.001, max(1e-6, min_oc * 0.001))
                    h = max_oc + intra_spread
                    l = max(min_oc * 0.999, min_oc - intra_spread)
                    l = max(1e-6, min(l, min_oc))
                    h = max(h, max_oc)
                    vwap = (o + c) / 2.0
                    vol = int(self.rng.integers(1_000_000, 5_000_000))
                else:
                    vol_est = 0.005 * max(c, 1e-6)
                    intra_h = abs(float(self.rng.normal(0.0, vol_est)))
                    intra_l = abs(float(self.rng.normal(0.0, vol_est)))
                    h = max_oc + intra_h
                    l = max(min_oc * 0.999, min_oc - intra_l)
                    l = max(1e-6, min(l, min_oc))
                    h = max(h, max_oc)
                    vwap = (o + 2.0 * c + h + l) / 5.0
                    base_vol = base_volumes.get(s, 10_000_000)
                    vol_noise = float(self.rng.uniform(0.8, 1.4))
                    vol = max(1000, int(base_vol * vol_noise))

                trade_count = max(50, int(vol / self.rng.uniform(80, 150)))

                # Ensure rounding respects sub-penny prices and strictly preserves low <= min(o, c) and high >= max(o, c)
                if min_oc >= 0.01:
                    r_o = round(o, 4)
                    r_c = round(c, 4)
                    r_h = round(h, 4)
                    r_l = round(l, 4)
                    r_vwap = round(vwap, 4)
                elif min_oc >= 0.0001:
                    r_o = round(o, 6)
                    r_c = round(c, 6)
                    r_h = round(h, 6)
                    r_l = round(l, 6)
                    r_vwap = round(vwap, 6)
                else:
                    r_o = max(1e-6, round(o, 8))
                    r_c = max(1e-6, round(c, 8))
                    r_h = max(1e-6, round(h, 8))
                    r_l = max(1e-6, round(l, 8))
                    r_vwap = max(1e-6, round(vwap, 8))

                r_min = min(r_o, r_c)
                r_max = max(r_o, r_c)
                if r_h < r_max:
                    r_h = r_max
                if r_l > r_min:
                    r_l = r_min
                if r_l <= 0.0:
                    r_l = max(1e-6, r_min * 0.999)

                bar = Bar(
                    symbol=s,
                    timestamp=bar_ts,
                    open=r_o,
                    high=r_h,
                    low=r_l,
                    close=r_c,
                    volume=vol,
                    trade_count=trade_count,
                    vwap=r_vwap,
                )
                result[s].append(bar)

            cur_date += timedelta(days=1)

        return result

    def generate_2008_crisis(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2008 Liquidity Crisis dataset."""
        sim = MertonJumpDiffusionSimulator(seed=seed)
        symbols = ["SPY", "QQQ", "TLT", "SHV", "GLD", "XLK", "XLE"]
        s0 = {"SPY": 140.0, "QQQ": 50.0, "TLT": 90.0, "SHV": 100.0, "GLD": 85.0, "XLK": 25.0, "XLE": 75.0}
        mu = {"SPY": -0.42, "QQQ": -0.45, "TLT": 0.28, "SHV": 0.02, "GLD": 0.05, "XLK": -0.50, "XLE": -0.40}
        sigma = {"SPY": 0.41, "QQQ": 0.44, "TLT": 0.19, "SHV": 0.001, "GLD": 0.25, "XLK": 0.45, "XLE": 0.45}
        lam = {"SPY": 8.0, "QQQ": 8.0, "TLT": 0.0, "SHV": 0.0, "GLD": 3.0, "XLK": 8.0, "XLE": 6.0}
        mu_j = {"SPY": -0.035, "QQQ": -0.035, "TLT": 0.0, "SHV": 0.0, "GLD": 0.01, "XLK": -0.035, "XLE": -0.03}
        sigma_j = {"SPY": 0.025, "QQQ": 0.025, "TLT": 0.0, "SHV": 0.0, "GLD": 0.02, "XLK": 0.025, "XLE": 0.02}

        corr = np.array([
            [1.00,  0.98, -0.45,  0.0,  0.10, 0.96, 0.85],
            [0.98,  1.00, -0.40,  0.0,  0.10, 0.98, 0.80],
            [-0.45, -0.40, 1.00,  0.0,  0.25,-0.40,-0.35],
            [0.00,  0.00,  0.00,  1.0,  0.00, 0.00, 0.00],
            [0.10,  0.10,  0.25,  0.0,  1.00, 0.10, 0.20],
            [0.96,  0.98, -0.40,  0.0,  0.10, 1.00, 0.82],
            [0.85,  0.80, -0.35,  0.0,  0.20, 0.82, 1.00],
        ], dtype=np.float64)

        paths = sim.simulate_multivariate_paths(
            symbols=symbols,
            initial_prices=s0,
            drifts=mu,
            volatilities=sigma,
            correlation_matrix=corr,
            jump_lambda=lam,
            jump_mean=mu_j,
            jump_vol=sigma_j,
            n_days=n_days,
        )
        return sim.paths_to_bars(paths, symbols, start_date=datetime(2008, 1, 2, tzinfo=timezone.utc))

    def generate_2020_flash_crash(self, n_days: int = 60, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2020 Flash Crash dataset: 23 days crash, (n_days - 23) days recovery."""
        sim = MertonJumpDiffusionSimulator(seed=seed)
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "TLT", "SHV", "GLD", "XLK", "XLE"]
        s0 = {"SPY": 330.0, "QQQ": 230.0, "AAPL": 80.0, "MSFT": 180.0, "TLT": 140.0, "SHV": 100.0, "GLD": 150.0, "XLK": 100.0, "XLE": 55.0}

        corr = np.array([
            # SPY   QQQ   AAPL  MSFT   TLT   SHV   GLD   XLK   XLE
            [1.00, 0.96, 0.94, 0.94, -0.25, 0.0, 0.15, 0.97, 0.85],
            [0.96, 1.00, 0.95, 0.95, -0.22, 0.0, 0.15, 0.99, 0.80],
            [0.94, 0.95, 1.00, 0.93, -0.20, 0.0, 0.10, 0.95, 0.80],
            [0.94, 0.95, 0.93, 1.00, -0.20, 0.0, 0.10, 0.95, 0.80],
            [-0.25,-0.22,-0.20,-0.20, 1.00, 0.0, 0.35,-0.25,-0.20],
            [0.00, 0.00, 0.00, 0.00,  0.00, 1.0, 0.00, 0.00, 0.00],
            [0.15, 0.15, 0.10, 0.10,  0.35, 0.0, 1.00, 0.15, 0.20],
            [0.97, 0.99, 0.95, 0.95, -0.25, 0.0, 0.15, 1.00, 0.82],
            [0.85, 0.80, 0.80, 0.80, -0.20, 0.0, 0.20, 0.82, 1.00],
        ], dtype=np.float64)

        crash_days = min(23, n_days)
        mu_c = {"SPY": -4.20, "QQQ": -3.90, "AAPL": -3.80, "MSFT": -3.60, "TLT": 0.45, "SHV": 0.01, "GLD": 0.10, "XLK": -3.80, "XLE": -4.50}
        sigma_c = {"SPY": 0.80, "QQQ": 0.82, "AAPL": 0.85, "MSFT": 0.82, "TLT": 0.28, "SHV": 0.002, "GLD": 0.35, "XLK": 0.82, "XLE": 0.95}

        paths_c = sim.simulate_multivariate_paths(
            symbols=symbols,
            initial_prices=s0,
            drifts=mu_c,
            volatilities=sigma_c,
            correlation_matrix=corr,
            n_days=crash_days,
        )

        recovery_days = n_days - crash_days
        if recovery_days > 0:
            s_rec = {s: paths_c[s][-1] for s in symbols}
            mu_r = {"SPY": 2.20, "QQQ": 2.80, "AAPL": 3.00, "MSFT": 2.80, "TLT": -0.15, "SHV": 0.01, "GLD": 0.25, "XLK": 2.90, "XLE": 1.20}
            sigma_r = {"SPY": 0.30, "QQQ": 0.32, "AAPL": 0.34, "MSFT": 0.32, "TLT": 0.16, "SHV": 0.002, "GLD": 0.20, "XLK": 0.30, "XLE": 0.45}

            paths_r = sim.simulate_multivariate_paths(
                symbols=symbols,
                initial_prices=s_rec,
                drifts=mu_r,
                volatilities=sigma_r,
                correlation_matrix=corr,
                n_days=recovery_days,
            )
            combined = {s: np.concatenate([paths_c[s], paths_r[s][1:]]) for s in symbols}
        else:
            combined = paths_c

        return sim.paths_to_bars(combined, symbols, start_date=datetime(2020, 2, 19, tzinfo=timezone.utc))

    def generate_2022_inflation_grind(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2022 Inflation Grind dataset: positive stock-bond correlation (+0.70)."""
        sim = MertonJumpDiffusionSimulator(seed=seed)
        symbols = ["SPY", "QQQ", "TLT", "SHV", "XLE", "GLD", "XLK"]
        s0 = {"SPY": 475.0, "QQQ": 400.0, "TLT": 145.0, "SHV": 100.0, "XLE": 55.0, "GLD": 170.0, "XLK": 175.0}
        mu = {"SPY": -0.16, "QQQ": -0.28, "TLT": 0.05, "SHV": 0.025, "XLE": 0.48, "GLD": 0.01, "XLK": -0.26}
        sigma = {"SPY": 0.22, "QQQ": 0.28, "TLT": 0.23, "SHV": 0.003, "XLE": 0.34, "GLD": 0.16, "XLK": 0.28}

        corr = np.array([
            # SPY    QQQ    TLT    SHV    XLE    GLD    XLK
            [ 1.00,  0.92,  0.70,  0.00,  0.30,  0.15,  0.95],
            [ 0.92,  1.00,  0.68,  0.00,  0.20,  0.12,  0.98],
            [ 0.70,  0.68,  1.00,  0.00,  0.05,  0.25,  0.70],
            [ 0.00,  0.00,  0.00,  1.00,  0.00,  0.00,  0.00],
            [ 0.30,  0.20,  0.05,  0.00,  1.00,  0.20,  0.25],
            [ 0.15,  0.12,  0.25,  0.00,  0.20,  1.00,  0.12],
            [ 0.95,  0.98,  0.70,  0.00,  0.25,  0.12,  1.00],
        ], dtype=np.float64)

        paths = sim.simulate_multivariate_paths(
            symbols=symbols,
            initial_prices=s0,
            drifts=mu,
            volatilities=sigma,
            correlation_matrix=corr,
            n_days=n_days,
        )
        return sim.paths_to_bars(paths, symbols, start_date=datetime(2022, 1, 3, tzinfo=timezone.utc))

    def generate_2017_low_vol_bull(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2017 Low-Vol Bull dataset: steady compounding, vol ~8%, max DD < 4%."""
        sim = MertonJumpDiffusionSimulator(seed=seed)
        symbols = ["SPY", "QQQ", "AAPL", "MSFT", "XLK", "SHV", "TLT", "GLD", "XLE"]
        s0 = {"SPY": 225.0, "QQQ": 120.0, "AAPL": 29.0, "MSFT": 62.0, "XLK": 48.0, "SHV": 100.0, "TLT": 120.0, "GLD": 110.0, "XLE": 68.0}
        mu = {"SPY": 0.22, "QQQ": 0.33, "AAPL": 0.38, "MSFT": 0.34, "XLK": 0.33, "SHV": 0.01, "TLT": 0.05, "GLD": 0.12, "XLE": -0.05}
        sigma = {"SPY": 0.075, "QQQ": 0.105, "AAPL": 0.135, "MSFT": 0.125, "XLK": 0.110, "SHV": 0.001, "TLT": 0.12, "GLD": 0.13, "XLE": 0.18}

        corr = np.array([
            # SPY   QQQ   AAPL  MSFT   XLK   SHV   TLT   GLD   XLE
            [1.00, 0.88, 0.82, 0.84, 0.90, 0.00,-0.35, 0.05, 0.60],
            [0.88, 1.00, 0.86, 0.88, 0.95, 0.00,-0.30, 0.05, 0.55],
            [0.82, 0.86, 1.00, 0.78, 0.88, 0.00,-0.25, 0.02, 0.45],
            [0.84, 0.88, 0.78, 1.00, 0.90, 0.00,-0.25, 0.05, 0.50],
            [0.90, 0.95, 0.88, 0.90, 1.00, 0.00,-0.30, 0.05, 0.58],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
            [-0.35,-0.30,-0.25,-0.25,-0.30, 0.00, 1.00, 0.30,-0.20],
            [0.05, 0.05, 0.02, 0.05, 0.05, 0.00, 0.30, 1.00, 0.15],
            [0.60, 0.55, 0.45, 0.50, 0.58, 0.00,-0.20, 0.15, 1.00],
        ], dtype=np.float64)

        paths = sim.simulate_multivariate_paths(
            symbols=symbols,
            initial_prices=s0,
            drifts=mu,
            volatilities=sigma,
            correlation_matrix=corr,
            n_days=n_days,
        )
        return sim.paths_to_bars(paths, symbols, start_date=datetime(2017, 1, 3, tzinfo=timezone.utc))
