"""Agents. Rule agents have per-agent random parameters and size trades as a fraction of their
own wealth (no clones). LLM agents decide once per step for all assets via a provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .llm import LLMClient
from .prompts import build_persona, build_user_prompt


@dataclass
class Order:
    ticker: str
    side: str       # BUY | SELL
    shares: float


class Agent:
    kind = "base"

    def __init__(self, agent_id: int, assets: List[str], cash: float, positions: Dict[str, float], max_trade_fraction: float):
        self.id = agent_id
        self.assets = assets
        self.cash = cash
        self.pos = dict(positions)
        self.max_trade_fraction = max_trade_fraction
        self.initial_value = None

    def value(self, mids: Dict[str, float]) -> float:
        return self.cash + sum(self.pos[a] * mids[a] for a in self.assets)

    def decide(self, market, news: Optional[str], step: int) -> List[Order]:
        raise NotImplementedError

    def _order_from_fraction(self, a: str, side: str, frac: float, mids: Dict[str, float]) -> Optional[Order]:
        """frac = fraction of total portfolio value to deploy (BUY) or of total value to liquidate (SELL)."""
        frac = float(np.clip(frac, 0.0, self.max_trade_fraction))
        if frac <= 0 or mids[a] <= 0:
            return None
        dollars = frac * self.value(mids)
        shares = dollars / mids[a]
        if side == "BUY":
            shares = min(shares, self.cash / mids[a] * 0.999)
        else:
            shares = min(shares, self.pos[a])
        if shares < 1e-6:
            return None
        return Order(a, side, shares)

    def execute(self, o: Order, market) -> bool:
        px = market.exec_price(o.ticker, o.side)
        if o.side == "BUY":
            cost = o.shares * px
            if cost > self.cash:
                return False
            self.cash -= cost
            self.pos[o.ticker] += o.shares
        else:
            if o.shares > self.pos[o.ticker] + 1e-9:
                return False
            self.cash += o.shares * px
            self.pos[o.ticker] -= o.shares
        market.record(o.ticker, o.side, o.shares, self.id)
        return True


class FundamentalAgent(Agent):
    kind = "fundamental"

    def __init__(self, *a, rng: np.random.Generator, **k):
        super().__init__(*a, **k)
        self.threshold = float(rng.uniform(0.005, 0.03))
        self.speed = float(rng.uniform(0.5, 2.0))

    def decide(self, market, news, step):
        mids = market.mids()
        out = []
        for a in self.assets:
            s = market.state[a]
            dev = (s.mid - s.fundamental) / s.fundamental
            if abs(dev) > self.threshold:
                frac = min(self.max_trade_fraction, self.speed * abs(dev))
                o = self._order_from_fraction(a, "SELL" if dev > 0 else "BUY", frac, mids)
                if o:
                    out.append(o)
        return out


class TrendAgent(Agent):
    kind = "trend"

    def __init__(self, *a, rng: np.random.Generator, **k):
        super().__init__(*a, **k)
        self.lookback = int(rng.integers(3, 11))
        self.threshold = float(rng.uniform(0.01, 0.04))
        self.frac = float(rng.uniform(0.01, 0.05))

    def decide(self, market, news, step):
        mids = market.mids()
        out = []
        for a in self.assets:
            r = market.sim_returns[a][-self.lookback:]
            if len(r) < self.lookback:
                continue
            cum = float(np.prod([1 + x for x in r]) - 1)
            if abs(cum) > self.threshold:
                o = self._order_from_fraction(a, "BUY" if cum > 0 else "SELL", self.frac, mids)
                if o:
                    out.append(o)
        return out


class NoiseAgent(Agent):
    kind = "noise"

    def __init__(self, *a, rng: np.random.Generator, **k):
        super().__init__(*a, **k)
        self.rng = rng
        self.prob = float(rng.uniform(0.05, 0.20))
        self.frac = float(rng.uniform(0.01, 0.05))

    def decide(self, market, news, step):
        if self.rng.random() > self.prob:
            return []
        a = str(self.rng.choice(self.assets))
        side = "BUY" if self.rng.random() < 0.5 else "SELL"
        o = self._order_from_fraction(a, side, self.frac, market.mids())
        return [o] if o else []


class LLMAgent(Agent):
    kind = "llm"

    def __init__(self, *a, client: LLMClient, provider: str, persona: str, prompt_variant: str = "free", **k):
        super().__init__(*a, **k)
        self.client = client
        self.provider = provider
        self.persona = persona
        self.system_prompt = build_persona(persona, prompt_variant)
        self.last_parse_ok: Optional[bool] = None

    def decide(self, market, news, step):
        mids = market.mids()
        snaps = [market.snapshot(a) for a in self.assets]
        user = build_user_prompt(snaps, self.pos, self.cash, self.value(mids), self.initial_value, news)
        res = self.client.decide(self.provider, self.system_prompt, user, meta={"agent": self.id, "step": step, "persona": self.persona})
        self.last_parse_ok = res.parsed is not None
        if res.parsed is None:
            return []
        out = []
        seen = set()
        trades = res.parsed.get("trades", [])
        if not isinstance(trades, list):
            return []
        for t in trades[: len(self.assets)]:
            if not isinstance(t, dict):
                continue
            a = str(t.get("ticker", "")).upper()
            if a not in self.assets or a in seen:
                continue
            seen.add(a)
            side = str(t.get("action", "HOLD")).upper()
            if side not in ("BUY", "SELL"):
                continue
            try:
                frac = float(t.get("fraction", 0.0))
            except (TypeError, ValueError):
                frac = 0.0
            o = self._order_from_fraction(a, side, frac, mids)
            if o:
                out.append(o)
        return out


def build_population(cfg, assets: List[str], initial_prices: Dict[str, float], share: float, arm: str,
                     client: Optional[LLMClient], seed: int) -> List[Agent]:
    """Seeded population. Same seed -> same wealth draws and rule parameters across arms/shares,
    so the only thing that changes with `share` is which agents are LLM-driven."""
    ac = cfg["agents"]
    n = int(ac["n_agents"])
    rng = np.random.default_rng(seed)
    wealth = np.maximum(5_000.0, rng.lognormal(np.log(ac["initial_cash"]), 0.8, size=n))
    rule_rng = np.random.default_rng(seed + 1)
    types = rule_rng.choice(["fundamental", "trend", "noise"], size=n, p=np.array(ac["rule_weights"]) / np.sum(ac["rule_weights"]))
    # LLM slots: a fixed random permutation; share=0.25 uses the first 25% of it, share=0.5 the first 50%, ...
    perm = np.random.default_rng(seed + 2).permutation(n)
    n_llm = int(round(n * share))
    llm_ids = set(perm[:n_llm].tolist())

    arm_cfg = cfg["llm"]["arms"][arm]
    providers = list(arm_cfg["providers"])
    personas = list(arm_cfg["personas"])
    variant = cfg["llm"].get("prompt_variant", "free")

    agents: List[Agent] = []
    eq = float(ac["equity_fraction"])
    k_llm = 0
    for i in range(n):
        w = float(wealth[i])
        positions = {a: (w * eq / len(assets)) / initial_prices[a] for a in assets}
        cash = w * (1 - eq)
        common = dict(agent_id=i, assets=assets, cash=cash, positions=positions, max_trade_fraction=float(ac["max_trade_fraction"]))
        if i in llm_ids:
            if client is None:
                raise RuntimeError("LLM share > 0 but no LLM client configured")
            ag = LLMAgent(**common, client=client, provider=providers[k_llm % len(providers)], persona=personas[k_llm % len(personas)], prompt_variant=variant)
            k_llm += 1
        else:
            t = types[i]
            cls = {"fundamental": FundamentalAgent, "trend": TrendAgent, "noise": NoiseAgent}[t]
            ag = cls(**common, rng=np.random.default_rng(seed * 1000 + i))
        ag.initial_value = w
        agents.append(ag)
    return agents
