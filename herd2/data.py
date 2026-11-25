"""Load CRSP CSVs and sample fixed episode windows (shared across all configs)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def load_prices(data_dir: str, assets: List[str]) -> pd.DataFrame:
    """Return DataFrame indexed by date with one adj_close column per asset."""
    cols = {}
    for a in assets:
        p = Path(data_dir) / f"{a.lower()}.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing data file {p}")
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        cols[a] = df["adj_close"].astype(float)
    px = pd.concat(cols, axis=1).dropna()
    px.columns = list(assets)
    return px


@dataclass
class Episode:
    episode_id: int
    scary: bool
    start_idx: int          # index into price frame of step 0
    dates: List[str]        # trading dates of the episode steps
    fundamental_returns: Dict[str, List[float]]   # per asset, per step (step 0 has return vs prior day)
    initial_prices: Dict[str, float]              # close on the day before step 0
    initial_vol: Dict[str, float]                 # annualised realised vol over lookback
    news_template_idx: int


def sample_episodes(px: pd.DataFrame, ep_cfg, assets: List[str]) -> List[Episode]:
    """Deterministic set of non-overlapping windows. Same for every arm/share."""
    rng = np.random.default_rng(int(ep_cfg["window_seed"]))
    n = int(ep_cfg["n_scary"]) + int(ep_cfg["n_placebo"])
    L = int(ep_cfg["length"])
    lb = int(ep_cfg["lookback"])
    lo, hi = pd.Timestamp(ep_cfg["date_range"][0]), pd.Timestamp(ep_cfg["date_range"][1])
    idx = px.index
    valid = np.where((idx >= lo) & (idx <= hi))[0]
    valid = valid[(valid >= lb + 1) & (valid + L < len(idx))]
    if len(valid) < n * L:
        raise ValueError("date_range too short for the requested number of episodes")

    starts: List[int] = []
    tries = 0
    while len(starts) < n and tries < 10000:
        tries += 1
        s = int(rng.choice(valid))
        if all(abs(s - t) >= L for t in starts):
            starts.append(s)
    starts.sort()
    scary_flags = np.array([True] * int(ep_cfg["n_scary"]) + [False] * int(ep_cfg["n_placebo"]))
    rng.shuffle(scary_flags)
    template_idx = rng.integers(0, 10_000, size=n)

    rets = px.pct_change()
    episodes = []
    for k, s in enumerate(starts):
        fr = {a: rets[a].iloc[s : s + L].tolist() for a in assets}
        # Every asset starts at 100: split-adjusted CRSP levels (NVDA at $0.68, AAPL at $24) made Haiku
        # distrust the data and also hint at the calendar date (look-ahead risk). Returns are unchanged;
        # impact/depth are defined relative to holdings, so results are invariant to this rescaling.
        init_px = {a: 100.0 for a in assets}
        init_vol = {a: float(rets[a].iloc[s - lb : s].std() * np.sqrt(252)) for a in assets}
        episodes.append(
            Episode(
                episode_id=k,
                scary=bool(scary_flags[k]),
                start_idx=s,
                dates=[d.strftime("%Y-%m-%d") for d in idx[s : s + L]],
                fundamental_returns=fr,
                initial_prices=init_px,
                initial_vol=init_vol,
                news_template_idx=int(template_idx[k]),
            )
        )
    return episodes
