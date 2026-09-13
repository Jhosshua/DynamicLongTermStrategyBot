"""Merton Jump-Diffusion Regime-Switching SDE Synthetic Market Simulator.
Provides calibrated stress datasets for 2008 Liquidity Crisis, 2020 Flash Crash,
2022 Inflation Grind, and 2017 Low-Vol Bull.
"""
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import numpy as np
from tests.mocks.contracts import Bar


class MertonJumpDiffusionSimulator:
    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)

    def simulate_multivariate_paths(
        self,
        symbols: List[str],
        initial_prices: Dict[str, float],
        drifts: Dict[str, float],       # Annualized mu
        volatilities: Dict[str, float], # Annualized sigma
        correlation_matrix: np.ndarray,
        jump_lambda: float = 0.0,       # Events per year
        jump_mean: float = 0.0,
        jump_vol: float = 0.0,
        n_days: int = 252,
        dt: float = 1.0 / 252.0,
        jump_sensitivities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        n_assets = len(symbols)
        v = np.array([volatilities[s] for s in symbols], dtype=float)
        cov = correlation_matrix * np.outer(v, v)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            cov += np.eye(n_assets) * 1e-6
            L = np.linalg.cholesky(cov)

        paths = {s: np.zeros(n_days + 1) for s in symbols}
        for s in symbols:
            paths[s][0] = initial_prices[s]

        # By default, cash equivalents (SHV, BIL) have 0 jump sensitivity
        if jump_sensitivities is None:
            jump_sensitivities = {
                s: (0.0 if s in ("SHV", "BIL", "CASH") else 1.0)
                for s in symbols
            }

        kappa = np.exp(jump_mean + 0.5 * (jump_vol ** 2)) - 1.0 if jump_lambda > 0 else 0.0

        for t in range(1, n_days + 1):
            z = self.rng.standard_normal(n_assets)
            diffusion = L @ z * np.sqrt(dt)
            n_jumps = self.rng.poisson(jump_lambda * dt) if jump_lambda > 0 else 0

            for i, s in enumerate(symbols):
                mu = drifts[s]
                jsens = jump_sensitivities.get(s, 1.0)
                drift_term = (mu - jsens * jump_lambda * kappa - 0.5 * (volatilities[s] ** 2)) * dt
                jump_term = 0.0
                if n_jumps > 0 and jsens > 0:
                    jump_term = jsens * np.sum(self.rng.normal(jump_mean, jump_vol, n_jumps))

                ret = drift_term + diffusion[i] + jump_term
                paths[s][t] = paths[s][t - 1] * np.exp(ret)

        return paths

    def generate_2008_crisis(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2008 Liquidity Crisis:
        SPY drops -55%, high vol (0.55-0.75), extreme negative jumps, TLT +28%, SHV +2%.
        """
        symbols = ["SPY", "QQQ", "TLT", "SHV", "GLD", "XLK", "XLE"]
        initial_prices = {"SPY": 140.0, "QQQ": 50.0, "TLT": 90.0, "SHV": 100.0, "GLD": 85.0, "XLK": 25.0, "XLE": 75.0}
        drifts = {"SPY": -0.65, "QQQ": -0.60, "TLT": 0.30, "SHV": 0.02, "GLD": 0.05, "XLK": -0.55, "XLE": -0.40}
        vols = {"SPY": 0.55, "QQQ": 0.58, "TLT": 0.20, "SHV": 0.002, "GLD": 0.25, "XLK": 0.52, "XLE": 0.48}
        corr = np.array([
            # SPY,  QQQ,   TLT,   SHV,  GLD,  XLK,  XLE
            [1.00,  0.95, -0.45,  0.0,  0.10, 0.96, 0.85],
            [0.95,  1.00, -0.40,  0.0,  0.10, 0.98, 0.80],
            [-0.45, -0.40, 1.00,  0.0,  0.25,-0.40,-0.35],
            [0.00,  0.00,  0.00,  1.0,  0.00, 0.00, 0.00],
            [0.10,  0.10,  0.25,  0.0,  1.00, 0.10, 0.20],
            [0.96,  0.98, -0.40,  0.0,  0.10, 1.00, 0.82],
            [0.85,  0.80, -0.35,  0.0,  0.20, 0.82, 1.00],
        ])
        sim = MertonJumpDiffusionSimulator(seed=seed)
        paths = sim.simulate_multivariate_paths(
            symbols, initial_prices, drifts, vols, corr,
            jump_lambda=12.0, jump_mean=-0.04, jump_vol=0.03, n_days=n_days
        )
        return self._paths_to_bars(paths, symbols, start_date=datetime(2008, 1, 2, tzinfo=timezone.utc))

    def generate_2020_flash_crash(self, n_days: int = 60, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2020 Flash Crash:
        Days 1-23: -35% crash, vol 85%, negative jumps.
        Days 24-60: +45% V-recovery led by QQQ.
        """
        symbols = ["SPY", "QQQ", "TLT", "SHV", "GLD", "XLK", "XLE"]
        initial_prices = {"SPY": 330.0, "QQQ": 230.0, "TLT": 140.0, "SHV": 100.0, "GLD": 150.0, "XLK": 100.0, "XLE": 55.0}
        corr = np.array([
            [1.00, 0.96, -0.30, 0.0, 0.15, 0.97, 0.85],
            [0.96, 1.00, -0.25, 0.0, 0.15, 0.99, 0.80],
            [-0.30,-0.25, 1.00, 0.0, 0.35,-0.25,-0.20],
            [0.00, 0.00,  0.00, 1.0, 0.00, 0.00, 0.00],
            [0.15, 0.15,  0.35, 0.0, 1.00, 0.15, 0.20],
            [0.97, 0.99, -0.25, 0.0, 0.15, 1.00, 0.82],
            [0.85, 0.80, -0.20, 0.0, 0.20, 0.82, 1.00],
        ])
        sim_crash = MertonJumpDiffusionSimulator(seed=seed)
        crash_days = 23
        paths_crash = sim_crash.simulate_multivariate_paths(
            symbols, initial_prices,
            drifts={"SPY": -2.50, "QQQ": -2.40, "TLT": 0.40, "SHV": 0.01, "GLD": 0.10, "XLK": -2.30, "XLE": -3.50},
            volatilities={"SPY": 0.85, "QQQ": 0.80, "TLT": 0.30, "SHV": 0.002, "GLD": 0.35, "XLK": 0.82, "XLE": 0.95},
            correlation_matrix=corr, jump_lambda=25.0, jump_mean=-0.05, jump_vol=0.03, n_days=crash_days
        )
        end_prices = {s: paths_crash[s][-1] for s in symbols}
        recover_days = n_days - crash_days
        sim_recover = MertonJumpDiffusionSimulator(seed=seed + 1)
        paths_recover = sim_recover.simulate_multivariate_paths(
            symbols, end_prices,
            drifts={"SPY": 1.80, "QQQ": 2.40, "TLT": -0.10, "SHV": 0.005, "GLD": 0.30, "XLK": 2.50, "XLE": 0.80},
            volatilities={"SPY": 0.30, "QQQ": 0.28, "TLT": 0.15, "SHV": 0.001, "GLD": 0.20, "XLK": 0.27, "XLE": 0.45},
            correlation_matrix=corr, jump_lambda=2.0, jump_mean=0.01, jump_vol=0.02, n_days=recover_days
        )
        combined = {}
        for s in symbols:
            combined[s] = np.concatenate([paths_crash[s], paths_recover[s][1:]])
        return self._paths_to_bars(combined, symbols, start_date=datetime(2020, 2, 19, tzinfo=timezone.utc))

    def generate_2022_inflation_grind(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2022 Inflation Grind:
        SPY -22%, TLT -31% (rate shock, positive correlation), XLE +55%, SHV +2.5%.
        """
        symbols = ["SPY", "QQQ", "TLT", "SHV", "GLD", "XLK", "XLE"]
        initial_prices = {"SPY": 475.0, "QQQ": 400.0, "TLT": 145.0, "SHV": 100.0, "GLD": 170.0, "XLK": 175.0, "XLE": 55.0}
        drifts = {"SPY": -0.22, "QQQ": -0.34, "TLT": -0.32, "SHV": 0.025, "GLD": -0.02, "XLK": -0.30, "XLE": 0.55}
        vols = {"SPY": 0.23, "QQQ": 0.29, "TLT": 0.24, "SHV": 0.003, "GLD": 0.18, "XLK": 0.28, "XLE": 0.35}
        corr = np.array([
            [1.00, 0.94,  0.68, 0.0, 0.10, 0.95, 0.40],
            [0.94, 1.00,  0.72, 0.0, 0.08, 0.98, 0.35],
            [0.68, 0.72,  1.00, 0.0, 0.20, 0.70, 0.10],
            [0.00, 0.00,  0.00, 1.0, 0.00, 0.00, 0.00],
            [0.10, 0.08,  0.20, 0.0, 1.00, 0.10, 0.25],
            [0.95, 0.98,  0.70, 0.0, 0.10, 1.00, 0.35],
            [0.40, 0.35,  0.10, 0.0, 0.25, 0.35, 1.00],
        ])
        sim = MertonJumpDiffusionSimulator(seed=seed)
        paths = sim.simulate_multivariate_paths(
            symbols, initial_prices, drifts, vols, corr,
            jump_lambda=1.0, jump_mean=-0.02, jump_vol=0.01, n_days=n_days
        )
        return self._paths_to_bars(paths, symbols, start_date=datetime(2022, 1, 3, tzinfo=timezone.utc))

    def generate_2017_low_vol_bull(self, n_days: int = 252, seed: int = 42) -> Dict[str, List[Bar]]:
        """Calibrated 2017 Low-Vol Bull:
        SPY +22%, QQQ +33%, low vol (8%), negligible jumps, max DD < 3%.
        """
        symbols = ["SPY", "QQQ", "TLT", "SHV", "GLD", "XLK", "XLE"]
        initial_prices = {"SPY": 225.0, "QQQ": 120.0, "TLT": 120.0, "SHV": 100.0, "GLD": 110.0, "XLK": 48.0, "XLE": 68.0}
        drifts = {"SPY": 0.22, "QQQ": 0.33, "TLT": 0.05, "SHV": 0.01, "GLD": 0.12, "XLK": 0.35, "XLE": -0.05}
        vols = {"SPY": 0.08, "QQQ": 0.11, "TLT": 0.12, "SHV": 0.001, "GLD": 0.13, "XLK": 0.11, "XLE": 0.18}
        corr = np.array([
            [1.00, 0.92, -0.35, 0.0, 0.05, 0.94, 0.60],
            [0.92, 1.00, -0.30, 0.0, 0.05, 0.97, 0.55],
            [-0.35,-0.30, 1.00, 0.0, 0.30,-0.30,-0.20],
            [0.00, 0.00,  0.00, 1.0, 0.00, 0.00, 0.00],
            [0.05, 0.05,  0.30, 0.0, 1.00, 0.05, 0.15],
            [0.94, 0.97, -0.30, 0.0, 0.05, 1.00, 0.58],
            [0.60, 0.55, -0.20, 0.0, 0.15, 0.58, 1.00],
        ])
        sim = MertonJumpDiffusionSimulator(seed=seed)
        paths = sim.simulate_multivariate_paths(
            symbols, initial_prices, drifts, vols, corr,
            jump_lambda=0.0, jump_mean=0.0, jump_vol=0.0, n_days=n_days
        )
        return self._paths_to_bars(paths, symbols, start_date=datetime(2017, 1, 3, tzinfo=timezone.utc))

    def _paths_to_bars(
        self,
        paths: Dict[str, np.ndarray],
        symbols: List[str],
        start_date: datetime,
    ) -> Dict[str, List[Bar]]:
        result = {s: [] for s in symbols}
        n_points = len(next(iter(paths.values())))

        cur_date = start_date
        for t in range(n_points):
            while cur_date.weekday() >= 5:
                cur_date += timedelta(days=1)

            for s in symbols:
                c = float(paths[s][t])
                # For SHV (cash proxy), tiny noise around close
                if s in ("SHV", "BIL", "CASH"):
                    o = c
                    high_val = c + 0.001
                    low_val = c - 0.001
                else:
                    vol_est = 0.008 * c
                    o = c * (1.0 + float(self.rng.normal(0, 0.002)))
                    high_val = max(o, c) + abs(float(self.rng.normal(0, vol_est)))
                    low_val = min(o, c) - abs(float(self.rng.normal(0, vol_est)))
                low_val = max(0.01, low_val)
                high_val = max(high_val, low_val + 0.001)
                vol = int(self.rng.integers(10_000, 100_000_000))
                bar = Bar(
                    symbol=s,
                    timestamp=cur_date,
                    open=round(o, 4),
                    high=round(high_val, 4),
                    low=round(low_val, 4),
                    close=round(c, 4),
                    volume=vol,
                    vwap=round((o + high_val + low_val + c) / 4.0, 4),
                )
                result[s].append(bar)

            cur_date += timedelta(days=1)
        return result
