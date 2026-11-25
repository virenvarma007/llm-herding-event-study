"""Run one 30-day episode and compute its metrics."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import news as newsmod
from .agents import LLMAgent, build_population
from .data import Episode
from .market import Market


def run_episode(cfg, ep: Episode, assets: List[str], share: float, arm: str, client, run_dir: Path, run_name: str) -> dict:
    ep_dir = run_dir / f"episode_{ep.episode_id:02d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"]) * 100 + ep.episode_id
    agents = build_population(cfg, assets, ep.initial_prices, share, arm, client, seed)
    llm_agents = [a for a in agents if isinstance(a, LLMAgent)]
    rule_agents = [a for a in agents if not isinstance(a, LLMAgent)]

    # depth relative to the population's aggregate holdings
    agg_shares = {a: sum(ag.pos[a] for ag in agents) for a in assets}
    base_depth = {a: cfg["market"]["depth_multiple"] * agg_shares[a] for a in assets}
    market = Market(assets, ep.initial_prices, ep.initial_vol, base_depth, cfg["market"])

    L = int(cfg["episodes"]["length"])
    E = int(cfg["episodes"]["event_day"])
    news_item = newsmod.pick(ep.scary, ep.news_template_idx)
    active = int(cfg["news"]["active_days"])
    rows = []                       # per-step aggregate action rows
    action_rows = []                # per-order rows
    llm_sell_frac, llm_buy_frac, rule_sell_frac, parse_fail = [], [], [], []
    pool = ThreadPoolExecutor(max_workers=int(cfg["llm"]["concurrency"])) if llm_agents else None
    t_start = time.time()
    stats0 = dict(client.stats) if client else {}
    stats_prev = dict(stats0)

    for step in range(L):
        news_txt = newsmod.format_for_llm(news_item, step - E) if E <= step < E + active else None
        # rule agents sequentially (cheap), LLM agents in parallel
        decisions: Dict[int, list] = {}
        for ag in rule_agents:
            decisions[ag.id] = ag.decide(market, None, step)
        if llm_agents:
            futs = {ag.id: pool.submit(ag.decide, market, news_txt, step) for ag in llm_agents}
            for aid, f in futs.items():
                decisions[aid] = f.result()
            client.ledger.flush(run_name)
            client.ledger.check()
            # circuit breaker: if most LLM calls in this step failed (bad key, bad model id, outage), stop now
            err_now = client.stats["errors"] - stats_prev["errors"]
            fail_now = client.stats["parse_fail"] - stats_prev["parse_fail"]
            stats_prev = dict(client.stats)
            # 0.2 (not 0.5): in the mixed arm one dead provider = exactly 50% errors, which `> 0.5` let through (2026-11-23 run)
            if err_now >= 0.2 * len(llm_agents) or fail_now >= 0.5 * len(llm_agents):
                last = _last_error(run_dir / "llm_calls.jsonl")
                raise RuntimeError(f"ABORT: step {step}: {err_now} errors / {fail_now} parse failures out of {len(llm_agents)} LLM calls. Last: {last}")
        # execute in a seeded random order so no agent type systematically trades first
        order = np.random.default_rng(seed * 1000 + step).permutation(len(agents))
        llm_sellers, llm_buyers, rule_sellers = set(), set(), set()
        for i in order:
            ag = agents[i]
            for o in decisions.get(ag.id, []):
                if ag.execute(o, market):
                    action_rows.append((step, ag.id, ag.kind, o.ticker, o.side, round(o.shares, 4)))
                    if ag.kind == "llm":
                        (llm_sellers if o.side == "SELL" else llm_buyers).add(ag.id)
                    elif o.side == "SELL":
                        rule_sellers.add(ag.id)
        market.step({a: ep.fundamental_returns[a][step] for a in assets})
        n_llm = max(1, len(llm_agents)); n_rule = max(1, len(rule_agents))
        llm_sell_frac.append(len(llm_sellers) / n_llm if llm_agents else np.nan)
        llm_buy_frac.append(len(llm_buyers) / n_llm if llm_agents else np.nan)
        rule_sell_frac.append(len(rule_sellers) / n_rule)
        pf = [a.last_parse_ok is False for a in llm_agents]
        parse_fail.append(float(np.mean(pf)) if pf else np.nan)
        rows.append({"step": step, "date": ep.dates[step], "news": bool(news_txt),
                     **{f"{a}_mid": market.state[a].mid for a in assets},
                     **{f"{a}_fund": market.state[a].fundamental for a in assets},
                     **{f"{a}_abn": market.abn_returns[a][-1] for a in assets},
                     **{f"{a}_spread_bps": market.spreads[a][-1] for a in assets},
                     **{f"{a}_depth_rel": market.depths[a][-1] for a in assets},
                     **{f"{a}_net_shares": market.net_shares[a][-1] for a in assets},
                     "llm_sell_frac": llm_sell_frac[-1], "llm_buy_frac": llm_buy_frac[-1], "rule_sell_frac": rule_sell_frac[-1]})
    if pool:
        pool.shutdown(wait=True)

    ts = pd.DataFrame(rows)
    ts.to_csv(ep_dir / "timeseries.csv", index=False)
    pd.DataFrame(action_rows, columns=["step", "agent_id", "kind", "ticker", "side", "shares"]).to_csv(ep_dir / "actions.csv", index=False)

    m = compute_metrics(ts, assets, E, llm_sell_frac, llm_buy_frac, rule_sell_frac, parse_fail)
    m.update({
        "episode_id": ep.episode_id, "scary": ep.scary, "start_date": ep.dates[0], "event_date": ep.dates[E],
        "headline": news_item["headline"], "arm": arm, "share": share, "n_llm": len(llm_agents), "n_agents": len(agents),
        "llm_calls": (client.stats["calls"] - stats0.get("calls", 0)) if client else 0,
        "llm_cost_usd": (client.stats["cost"] - stats0.get("cost", 0.0)) if client else 0.0,
        "llm_errors": (client.stats["errors"] - stats0.get("errors", 0)) if client else 0,
        "wall_s": round(time.time() - t_start, 1),
    })
    with open(ep_dir / "metrics.json", "w") as f:
        json.dump(m, f, indent=2, default=float)
    return m


def _last_error(log_path: Path) -> str:
    try:
        lines = log_path.read_text().splitlines()[-200:]
        for ln in reversed(lines):
            r = json.loads(ln)
            if r.get("error"):
                return str(r["error"])[:300]
            if r.get("parsed") is None:
                return "unparsed content: " + str(r.get("content"))[:200]
    except Exception:
        pass
    return "n/a"


def compute_metrics(ts: pd.DataFrame, assets: List[str], E: int, llm_sell, llm_buy, rule_sell, parse_fail) -> dict:
    """Event-study metrics. CAR_k = cumulative abnormal log return from day E-1 close through day E+k."""
    out = {}
    abn = {a: ts[f"{a}_abn"].to_numpy() for a in assets}
    for k in (0, 1, 2, 5, 10, 19):
        hi = min(len(ts) - 1, E + k)
        out[f"car_spy_d{k}"] = float(np.sum(abn["SPY"][E : hi + 1]))
        out[f"car_mean_d{k}"] = float(np.mean([np.sum(abn[a][E : hi + 1]) for a in assets]))
    # pre-event drift (should be ~0 and equal across configs)
    out["car_spy_pre"] = float(np.sum(abn["SPY"][max(0, E - 5) : E]))
    # minimum CAR within 10 days after event (max abnormal drawdown)
    cum = np.cumsum(abn["SPY"][E : E + 11])
    out["min_car_spy_d10"] = float(np.min(cum)) if len(cum) else 0.0
    # selling synchronicity (unique agents / agents of that kind)
    def win(x, lo, hi):
        v = [t for t in x[max(0, lo) : hi + 1] if not (isinstance(t, float) and np.isnan(t))]
        return float(np.mean(v)) if v else float("nan")
    out["llm_sell_frac_pre"] = win(llm_sell, E - 5, E - 1)
    out["llm_sell_frac_d0_2"] = win(llm_sell, E, E + 2)
    out["llm_sell_frac_d0_9"] = win(llm_sell, E, E + 9)
    out["llm_sell_frac_peak_d0_9"] = float(np.nanmax(llm_sell[E : E + 10])) if llm_sell and not all(np.isnan(llm_sell[E:E+10])) else float("nan")
    out["llm_buy_frac_d0_2"] = win(llm_buy, E, E + 2)
    out["rule_sell_frac_pre"] = win(rule_sell, E - 5, E - 1)
    out["rule_sell_frac_d0_2"] = win(rule_sell, E, E + 2)
    # liquidity: relative change in SPY spread and depth, post (d0..d5) vs pre (d-5..d-1)
    for a in ("SPY",):
        sp, dp = ts[f"{a}_spread_bps"].to_numpy(), ts[f"{a}_depth_rel"].to_numpy()
        out["spread_change_d0_5"] = float(np.mean(sp[E : E + 6]) / np.mean(sp[max(0, E - 5) : E]) - 1)
        out["depth_change_d0_5"] = float(np.mean(dp[E : E + 6]) / np.mean(dp[max(0, E - 5) : E]) - 1)
    # cross-asset co-movement of ABNORMAL returns, post window
    A = np.array([abn[a][E : E + 20] for a in assets])
    if A.shape[1] >= 5 and np.all(np.std(A, axis=1) > 1e-12):
        c = np.corrcoef(A); n = len(assets)
        out["abn_corr_post"] = float((c.sum() - n) / (n * (n - 1)))
    else:
        out["abn_corr_post"] = float("nan")
    out["parse_fail_rate"] = float(np.nanmean(parse_fail)) if parse_fail and not all(np.isnan(parse_fail)) else float("nan")
    return out
