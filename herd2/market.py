"""Price-impact market. Depth is defined relative to the agent population's own holdings,
so the impact of a coordinated sell-off is interpretable and independent of CRSP volume."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AssetState:
    mid: float
    fundamental: float
    spread: float
    depth: float
    vol: float
    buy_shares: float = 0.0
    sell_shares: float = 0.0
    buyers: set = field(default_factory=set)
    sellers: set = field(default_factory=set)


class Market:
    def __init__(self, assets: List[str], initial_prices: Dict[str, float], initial_vol: Dict[str, float],
                 base_depth_shares: Dict[str, float], cfg):
        self.assets = assets
        self.cfg = cfg
        self.base_depth = dict(base_depth_shares)
        self.state: Dict[str, AssetState] = {}
        for a in assets:
            p = initial_prices[a]
            self.state[a] = AssetState(
                mid=p, fundamental=p, spread=p * cfg["base_spread_bps"] / 1e4,
                depth=self.base_depth[a], vol=max(0.05, initial_vol.get(a, 0.2)),
            )
        self.sim_returns: Dict[str, List[float]] = {a: [] for a in assets}
        self.fund_returns: Dict[str, List[float]] = {a: [] for a in assets}
        self.abn_returns: Dict[str, List[float]] = {a: [] for a in assets}   # log(1+sim) - log(1+fund)
        self.spreads: Dict[str, List[float]] = {a: [] for a in assets}
        self.depths: Dict[str, List[float]] = {a: [] for a in assets}
        self.net_shares: Dict[str, List[float]] = {a: [] for a in assets}
        self.n_sellers: Dict[str, List[int]] = {a: [] for a in assets}
        self.n_buyers: Dict[str, List[int]] = {a: [] for a in assets}

    # ---- order handling -------------------------------------------------
    def exec_price(self, a: str, side: str) -> float:
        s = self.state[a]
        return s.mid + s.spread / 2 if side == "BUY" else s.mid - s.spread / 2

    def record(self, a: str, side: str, shares: float, agent_id: int) -> None:
        s = self.state[a]
        if side == "BUY":
            s.buy_shares += shares
            s.buyers.add(agent_id)
        else:
            s.sell_shares += shares
            s.sellers.add(agent_id)

    def snapshot(self, a: str, lookback: int = 5) -> dict:
        s = self.state[a]
        r = self.sim_returns[a][-lookback:]
        return {
            "ticker": a, "mid": s.mid, "spread_bps": s.spread / s.mid * 1e4, "vol": s.vol,
            "ret_1d": r[-1] if r else 0.0, "ret_5d": float(np.prod([1 + x for x in r]) - 1) if r else 0.0,
            "fundamental": s.fundamental,
        }

    # ---- end of day -----------------------------------------------------
    def step(self, fundamental_returns: Dict[str, float]) -> None:
        c = self.cfg
        for a in self.assets:
            s = self.state[a]
            fr = float(fundamental_returns[a])
            net = s.buy_shares - s.sell_shares
            total = s.buy_shares + s.sell_shares
            imb_norm = net / total if total > 0 else 0.0

            liquidity = self.base_depth[a] / (1 + c["vol_sensitivity"] * s.vol)
            liquidity = liquidity / (1 + c["imbalance_sensitivity"] * abs(imb_norm))
            liquidity = max(0.1 * self.base_depth[a], liquidity)

            impact = c["impact_alpha"] * net / liquidity
            impact = float(np.clip(impact, -c["max_daily_impact"], c["max_daily_impact"]))
            sim_r = (1 + fr) * (1 + impact) - 1

            s.fundamental *= 1 + fr
            s.mid *= 1 + sim_r
            s.spread = s.mid * c["base_spread_bps"] / 1e4 * (1 + c["vol_sensitivity"] * s.vol) * (1 + c["imbalance_sensitivity"] * abs(imb_norm))
            s.depth = liquidity

            self.sim_returns[a].append(sim_r)
            self.fund_returns[a].append(fr)
            self.abn_returns[a].append(float(np.log1p(sim_r) - np.log1p(fr)))
            self.spreads[a].append(s.spread / s.mid * 1e4)
            self.depths[a].append(liquidity / self.base_depth[a])
            self.net_shares[a].append(net)
            self.n_sellers[a].append(len(s.sellers))
            self.n_buyers[a].append(len(s.buyers))

            # rolling vol of simulated returns (20d), blended with initial
            rr = self.sim_returns[a][-20:]
            if len(rr) >= 5:
                s.vol = float(np.std(rr) * np.sqrt(252)) * min(1.0, len(rr) / 20) + s.vol * (1 - min(1.0, len(rr) / 20))
            s.buy_shares = s.sell_shares = 0.0
            s.buyers, s.sellers = set(), set()

    def mids(self) -> Dict[str, float]:
        return {a: self.state[a].mid for a in self.assets}
