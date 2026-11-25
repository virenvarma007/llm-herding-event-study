"""CLI.

  python -m herd2.run plan                         # calls + cost estimate for the full sweep
  python -m herd2.run check-models                 # 1 tiny call per provider: verifies keys, model ids, prices
  python -m herd2.run run --arm mono --share 0.5   # one config, all episodes (resumable)
  python -m herd2.run run --arm mono --share 0.5 --provider mock --episodes 2   # pipeline test, no API calls
  python -m herd2.run sweep                        # every arm x share, sequentially in this process
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config, load_dotenv, run_name
from .data import load_prices, sample_episodes
from .episode import run_episode
from .llm import LLMClient


def _episodes(cfg):
    px = load_prices(cfg["data_dir"], cfg["assets"])
    return sample_episodes(px, cfg["episodes"], cfg["assets"])


def cmd_plan(cfg, args):
    eps = _episodes(cfg)
    n = int(cfg["agents"]["n_agents"]); L = int(cfg["episodes"]["length"])
    tok_in, tok_out = 450, 80
    pa = cfg["llm"]["providers"]["anthropic"]
    print(f"episodes: {len(eps)} ({sum(e.scary for e in eps)} scary, {sum(not e.scary for e in eps)} placebo), {L} steps each")
    print("windows:", ", ".join(f"{e.dates[0]}{'*' if e.scary else ''}" for e in eps), "(* = scary)")
    total_calls = 0; total_anth = 0.0; total_or = 0.0
    for arm in cfg["sweep"]["arms"]:
        provs = cfg["llm"]["arms"][arm]["providers"]
        for share in cfg["sweep"]["shares"]:
            calls = int(round(n * share)) * L * len(eps)
            anth_share = sum(1 for p in provs if cfg["llm"]["providers"][p]["kind"] == "anthropic") / len(provs)
            c_anth = calls * anth_share * (tok_in * pa["price_in_per_m"] + tok_out * pa["price_out_per_m"]) / 1e6
            c_or = calls * (1 - anth_share) * (tok_in * 0.30 + tok_out * 1.20) / 1e6   # rough OpenRouter guess
            total_calls += calls; total_anth += c_anth; total_or += c_or
            print(f"  {arm:8s} share {share:.2f}: {calls:7d} calls  ~${c_anth:5.2f} Claude  ~${c_or:5.2f} OpenRouter")
    print(f"TOTAL: {total_calls} calls, ~${total_anth:.2f} Claude + ~${total_or:.2f} OpenRouter")
    print("OpenRouter figure assumes $0.30/$1.20 per M; `check-models` prints the real price of the configured model. Thinking/reasoning is off.")


def cmd_check_models(cfg, args):
    load_dotenv()
    from .prompts import PERSONAS, build_user_prompt
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_check"; tmp.mkdir(exist_ok=True)
    client = LLMClient(cfg, tmp, "check")
    client.cache = None
    snaps = [{"ticker": a, "mid": 100.0, "ret_1d": -0.012, "ret_5d": -0.03, "vol": 0.22, "spread_bps": 6.1} for a in cfg["assets"]]
    user = build_user_prompt(snaps, {a: 50.0 for a in cfg["assets"]}, 50000.0, 75000.0, 75000.0,
                             "NEWS (TODAY):\nHeadline: Unconfirmed Reports of Major Bank Liquidity Strain\nUnverified reports of an acute funding shortfall at a large bank.")
    try:
        prices = client.load_openrouter_prices()
    except Exception as e:
        prices = {}; print("could not fetch OpenRouter price list:", e)
    ok = True
    for name, pc in cfg["llm"]["providers"].items():
        t0 = time.time()
        r = client.decide(name, PERSONAS["baseline"], user, meta={"check": True})
        dt = time.time() - t0
        status = "OK " if r.parsed is not None else "FAIL"
        ok &= r.parsed is not None
        pr = prices.get(pc["model"])
        price_s = f"${pr['prompt']*1e6:.3f}/{pr['completion']*1e6:.3f} per M" if pr else (f"${pc.get('price_in_per_m')}/{pc.get('price_out_per_m')} per M" if pc.get("price_in_per_m") else "price: unknown")
        print(f"[{status}] {name:13s} {pc['model']:38s} {dt:5.1f}s  in={r.usage_in} out={r.usage_out}  cost=${r.cost_usd:.5f}  {price_s}")
        if r.error:
            print("       error:", r.error)
        elif r.parsed is None:
            print("       unparsed content:", (r.content or "")[:200])
        else:
            print("       decision:", json.dumps(r.parsed)[:200])
    client.close()
    print("ALL OK" if ok else "SOME PROVIDERS FAILED -- fix config.yaml model ids / keys before running")
    return 0 if ok else 1


def cmd_run(cfg, args, arm=None, share=None):
    load_dotenv()
    arm = arm or args.arm; share = float(share if share is not None else args.share)
    provider = args.provider
    name = run_name(arm, share, provider, cfg["llm"].get("prompt_variant", "free"))
    run_dir = Path(cfg["out_dir"]) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    eps = _episodes(cfg)
    if args.episodes and args.episodes != "all":
        eps = eps[: int(args.episodes)]
    with open(run_dir / "config_resolved.json", "w") as f:
        json.dump({"arm": arm, "share": share, "provider_override": provider, "cfg": cfg}, f, indent=2, default=str)
    client = LLMClient(cfg, run_dir, name, provider_override=provider) if share > 0 else None
    if client and provider is None:
        try:
            client.load_openrouter_prices()
        except Exception:
            pass
    print(f"== {name}: {len(eps)} episodes, {int(round(cfg['agents']['n_agents'] * share))} LLM agents, arm={arm}", flush=True)
    done = 0
    for ep in eps:
        mpath = run_dir / f"episode_{ep.episode_id:02d}" / "metrics.json"
        if mpath.exists() and not args.force:
            done += 1
            continue
        m = run_episode(cfg, ep, cfg["assets"], share, arm, client, run_dir, name)
        done += 1
        st = client.stats if client else {}
        print(f"  ep {ep.episode_id:02d} {'SCARY  ' if ep.scary else 'placebo'} {ep.dates[0]}  CAR_spy_d5={m['car_spy_d5']:+.4f}  "
              f"llm_sell_d0-2={m['llm_sell_frac_d0_2']:.2f} rule_sell_d0-2={m['rule_sell_frac_d0_2']:.2f}  "
              f"calls={st.get('calls',0)} cached={st.get('cached',0)} parse_fail={st.get('parse_fail',0)} err={st.get('errors',0)} "
              f"${st.get('cost',0.0):.2f}  {m['wall_s']}s", flush=True)
    if client:
        client.ledger.flush(name)
        with open(run_dir / "llm_stats.json", "w") as f:
            json.dump(client.stats, f, indent=2)
        client.close()
    print(f"== {name} done ({done} episodes)", flush=True)


def cmd_sweep(cfg, args):
    for arm in cfg["sweep"]["arms"]:
        for share in cfg["sweep"]["shares"]:
            if share == 0.0 and arm != cfg["sweep"]["arms"][0]:
                continue   # share 0 is the same control for every arm; run it once
            cmd_run(cfg, args, arm=arm, share=share)


def main(argv=None):
    p = argparse.ArgumentParser(prog="herd2")
    p.add_argument("--config", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("check-models")
    for name in ("run", "sweep"):
        s = sub.add_parser(name)
        s.add_argument("--arm", default="mono")
        s.add_argument("--share", type=float, default=0.5)
        s.add_argument("--episodes", default="all", help="all or an integer (first k episodes)")
        s.add_argument("--provider", default=None, help="override provider kind, e.g. mock")
        s.add_argument("--force", action="store_true", help="re-run finished episodes")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    if args.cmd == "plan":
        return cmd_plan(cfg, args)
    if args.cmd == "check-models":
        return cmd_check_models(cfg, args)
    if args.cmd == "run":
        return cmd_run(cfg, args)
    if args.cmd == "sweep":
        return cmd_sweep(cfg, args)


if __name__ == "__main__":
    sys.exit(main() or 0)
