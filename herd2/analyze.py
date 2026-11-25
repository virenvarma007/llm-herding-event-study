"""Aggregate all runs -> tables, bootstrap CIs, clustered regression, plots.

  python -m herd2.analyze                 # reads out/*/episode_*/metrics.json
  python -m herd2.analyze --out out_mock  # analyze a mock sweep
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

KEY = ["car_spy_d2", "car_spy_d5", "car_spy_d10", "car_mean_d5", "min_car_spy_d10",
       "llm_sell_frac_d0_2", "llm_sell_frac_peak_d0_9", "rule_sell_frac_d0_2",
       "spread_change_d0_5", "depth_change_d0_5", "abn_corr_post", "parse_fail_rate"]


def load_all(out_dir: str) -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(Path(out_dir) / "*" / "episode_*" / "metrics.json")):
        with open(p) as f:
            m = json.load(f)
        m["run"] = Path(p).parent.parent.name
        m["mock"] = m["run"].endswith("_mock")
        rows.append(m)
    if not rows:
        raise SystemExit(f"no metrics.json under {out_dir}")
    df = pd.DataFrame(rows)
    df["scary"] = df["scary"].astype(bool)
    return df


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (arm, share, scary), g in df.groupby(["arm", "share", "scary"]):
        row = {"arm": arm, "share": share, "scary": scary, "n_episodes": len(g)}
        for k in KEY:
            if k in g:
                v = g[k].to_numpy(dtype=float)
                lo, hi = boot_ci(v)
                row[k] = float(np.nanmean(v)); row[f"{k}_lo"] = lo; row[f"{k}_hi"] = hi
        out.append(row)
    return pd.DataFrame(out).sort_values(["scary", "arm", "share"], ascending=[False, True, True])


def paired_effects(df: pd.DataFrame) -> pd.DataFrame:
    """Same episode under share s vs share 0 (paired by episode_id): treatment effect per episode."""
    ctrl = df[df["share"] == 0.0].set_index("episode_id")
    if ctrl.empty:
        return pd.DataFrame()
    rows = []
    for (arm, share), g in df[df["share"] > 0].groupby(["arm", "share"]):
        g = g.set_index("episode_id")
        common = g.index.intersection(ctrl.index)
        for scary in (True, False):
            idx = [i for i in common if bool(g.loc[i, "scary"]) == scary]
            if not idx:
                continue
            for k in ("car_spy_d2", "car_spy_d5", "car_spy_d10", "min_car_spy_d10", "spread_change_d0_5", "abn_corr_post"):
                d = (g.loc[idx, k].to_numpy(dtype=float) - ctrl.loc[idx, k].to_numpy(dtype=float))
                lo, hi = boot_ci(d)
                rows.append({"arm": arm, "share": share, "scary": scary, "metric": k, "n": len(idx),
                             "effect_vs_share0": float(np.nanmean(d)), "ci_lo": lo, "ci_hi": hi,
                             "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows)


def regression(df: pd.DataFrame, y: str = "car_spy_d5") -> str:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels not installed; skipping regression"
    d = df.copy()
    d["scary_i"] = d["scary"].astype(int)
    d = d.dropna(subset=[y])
    # share 0 belongs to every arm (it is the shared control); duplicate it per arm for the arm interactions
    arms = sorted(a for a in d["arm"].unique())
    ctrl = d[d["share"] == 0.0]
    d = pd.concat([d[d["share"] > 0]] + [ctrl.assign(arm=a) for a in arms], ignore_index=True)
    res = []
    for formula in (f"{y} ~ share * scary_i", f"{y} ~ share * scary_i * C(arm, Treatment('mono'))"):
        try:
            m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["episode_id"]})
            res.append(f"--- OLS, SE clustered by episode ---\n{formula}\n{m.summary().tables[1]}\nR2={m.rsquared:.3f}  n={int(m.nobs)}")
        except Exception as e:
            res.append(f"{formula}: failed ({e})")
    return "\n\n".join(res)


def plots(df: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted(df["arm"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (k, title) in zip(axes, [("car_spy_d5", "SPY abnormal return, days 0-5"),
                                      ("llm_sell_frac_d0_2", "LLM agents selling, days 0-2"),
                                      ("spread_change_d0_5", "SPY spread change, days 0-5")]):
        for arm in arms:
            for scary, ls in ((True, "-"), (False, "--")):
                g = df[(df["scary"] == scary) & ((df["arm"] == arm) | (df["share"] == 0.0))]
                s = g.groupby("share")[k].agg(["mean"])
                ci = g.groupby("share")[k].apply(lambda v: boot_ci(v.to_numpy(dtype=float)))
                lo = np.array([c[0] for c in ci]); hi = np.array([c[1] for c in ci])
                ax.plot(s.index, s["mean"], ls, marker="o", label=f"{arm} {'scary' if scary else 'placebo'}")
                ax.fill_between(s.index, lo, hi, alpha=0.12)
        ax.set_title(title); ax.set_xlabel("LLM share"); ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_dose_response.png", dpi=150); plt.close(fig)

    # CAR paths (SPY) by share, scary episodes only, averaged over episodes
    fig, axes = plt.subplots(1, len(arms), figsize=(5 * len(arms), 4), squeeze=False)
    for ax, arm in zip(axes[0], arms):
        for run_dir in sorted(glob.glob(str(out_dir / f"{arm}_share*"))) + sorted(glob.glob(str(out_dir / "*_share0.00*"))):
            paths = []
            for ts_path in glob.glob(f"{run_dir}/episode_*/timeseries.csv"):
                mp = Path(ts_path).parent / "metrics.json"
                if not mp.exists() or not json.load(open(mp))["scary"]:
                    continue
                ts = pd.read_csv(ts_path)
                E = ts.index[ts["news"]].min() if ts["news"].any() else 10
                paths.append(np.cumsum(ts["SPY_abn"].to_numpy()[E - 5 : E + 20]) - np.sum(ts["SPY_abn"].to_numpy()[E - 5 : E]))
            if paths:
                P = np.array(paths); x = np.arange(-5, -5 + P.shape[1])
                ax.plot(x, P.mean(0), label=Path(run_dir).name.replace(f"{arm}_", "").replace("mono_", ""))
        ax.axvline(0, color="r", lw=0.6); ax.axhline(0, color="k", lw=0.5); ax.set_title(f"{arm}: mean SPY CAR path (scary)")
        ax.set_xlabel("days from rumor"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out_dir / "fig_car_paths.png", dpi=150); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="output dir to analyze (default: config out_dir)")
    p.add_argument("--include-mock", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = Path(args.out or cfg["out_dir"])
    df = load_all(str(out_dir))
    if not args.include_mock:
        df = df[~df["mock"]]
        if df.empty:
            raise SystemExit("only mock runs found; pass --include-mock to analyze them")
    df.to_csv(out_dir / "all_episodes.csv", index=False)
    summ = summary_table(df); summ.to_csv(out_dir / "summary_by_config.csv", index=False)
    eff = paired_effects(df); eff.to_csv(out_dir / "paired_effects_vs_share0.csv", index=False)
    reg = regression(df)
    (out_dir / "regression.txt").write_text(reg)
    plots(df, out_dir)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("== episodes per config ==")
    print(df.groupby(["arm", "share", "scary"]).size().unstack(fill_value=0))
    print("\n== means (scary episodes) ==")
    cols = ["arm", "share", "n_episodes", "car_spy_d5", "car_spy_d5_lo", "car_spy_d5_hi", "llm_sell_frac_d0_2", "rule_sell_frac_d0_2", "spread_change_d0_5", "abn_corr_post", "parse_fail_rate"]
    print(summ[summ["scary"]][[c for c in cols if c in summ]].round(4).to_string(index=False))
    print("\n== means (placebo episodes) ==")
    print(summ[~summ["scary"]][[c for c in cols if c in summ]].round(4).to_string(index=False))
    if not eff.empty:
        print("\n== paired effect vs share 0 (CAR SPY d5) ==")
        print(eff[eff["metric"] == "car_spy_d5"].round(4).to_string(index=False))
    print("\n" + reg)
    print(f"\nwrote {out_dir}/summary_by_config.csv, paired_effects_vs_share0.csv, regression.txt, fig_dose_response.png, fig_car_paths.png")


if __name__ == "__main__":
    main()
