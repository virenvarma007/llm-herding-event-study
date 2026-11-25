"""Paper figures (static PDF for arXiv). Run: .venv/bin/python -m herd2.paper_figs"""
import glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("out"); FIG = Path("paper/figs"); FIG.mkdir(parents=True, exist_ok=True)
COL = {"mono": "#2a78d6", "persona": "#eb6834", "mixed": "#1baf7a", "control": "#52514e"}
LABEL = {"mono": "mono (Haiku)", "persona": "persona (Haiku, 5 prompts)", "mixed": "mixed (Haiku + DeepSeek)"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"], "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "legend.frameon": False})
rng = np.random.default_rng(0)

def boot(v, n=4000):
    v = np.asarray(v, float); m = v.mean()
    b = rng.choice(v, (n, len(v))).mean(1)
    return m, np.percentile(b, 2.5), np.percentile(b, 97.5)

df = pd.read_csv(OUT / "all_episodes.csv"); df["scary"] = df["scary"].astype(bool)
ctrl = df[df.share == 0].drop_duplicates("episode_id").set_index("episode_id")
for c in ["car_spy_d5", "car_spy_d2", "car_spy_d10"]:
    df["d_" + c] = df[c].values - ctrl[c].reindex(df.episode_id).values

# ---- Fig 1: dose-response -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6), sharey=True)
for ax, scary, ttl in zip(axes, [True, False], ["Scary rumor (n = 24 episodes)", "Placebo news (n = 12 episodes)"]):
    for arm in ["mono", "persona", "mixed"]:
        xs, ms, los, his = [0.0], [0.0], [0.0], [0.0]
        for sh in [0.25, 0.5, 1.0]:
            v = df[(df.arm == arm) & (df.share == sh) & (df.scary == scary)].d_car_spy_d5
            m, lo, hi = boot(v); xs.append(sh); ms.append(m); los.append(lo); his.append(hi)
        ax.plot(xs, np.array(ms) * 100, "-o", color=COL[arm], lw=2, ms=5, label=LABEL[arm])
        ax.fill_between(xs, np.array(los) * 100, np.array(his) * 100, color=COL[arm], alpha=0.15, lw=0)
        if scary:
            ax.annotate(f"{ms[-1]*100:+.1f}%", (1.0, ms[-1] * 100), xytext=(4, 0), textcoords="offset points", fontsize=7, va="center", color="#0b0b0b")
    ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks([0, 0.25, 0.5, 1.0]); ax.set_xlabel("share of LLM-driven agents")
    ax.set_title(ttl, fontsize=9); ax.set_xlim(-0.03, 1.15)
axes[0].set_ylabel("SPY abnormal return, days 0–5,\nvs. share-0 control (pp)")
axes[0].legend(loc="lower left", fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig1_dose_response.pdf"); fig.savefig(FIG / "fig1_dose_response.png", dpi=160); plt.close(fig)

# ---- Fig 2: event-time CAR paths (scary), by arm, lines by share ---------
def car_paths(run_dir, scary=True):
    paths = []
    for ts_path in glob.glob(f"{run_dir}/episode_*/timeseries.csv"):
        mp = Path(ts_path).parent / "metrics.json"
        if not mp.exists() or json.load(open(mp))["scary"] != scary: continue
        ts = pd.read_csv(ts_path); a = ts["SPY_abn"].to_numpy(); E = 10
        paths.append(np.cumsum(a[E - 5: E + 20]) - a[E - 5: E].sum())
    return np.array(paths)
SHADE = {0.25: 0.35, 0.5: 0.65, 1.0: 1.0}
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.3), sharey=True)
ctrl_paths = car_paths(OUT / "mono_share0.00")
x = np.arange(-5, 20)
for ax, arm in zip(axes, ["mono", "persona", "mixed"]):
    ax.axvspan(0, 9, color="#eeeeea", lw=0, zorder=0); ax.axhline(0, color="#52514e", lw=0.8)
    ax.plot(x, ctrl_paths.mean(0) * 100, color=COL["control"], lw=1.2, ls="--", label="control (share 0)")
    for sh, tag in [(0.25, "0.25"), (0.5, "0.50"), (1.0, "1.00")]:
        P = car_paths(OUT / f"{arm}_share{tag}")
        ax.plot(x, P.mean(0) * 100, color=COL[arm], lw=2, alpha=SHADE[sh], label=f"share {sh}")
    ax.set_title(LABEL[arm], fontsize=9); ax.set_xlabel("trading days from rumor")
axes[0].set_ylabel("mean SPY CAR (pp)"); axes[0].legend(fontsize=7, loc="lower left")
axes[1].text(4.5, axes[1].get_ylim()[1] * 0.92, "news shown", ha="center", fontsize=7, color="#52514e")
fig.tight_layout(); fig.savefig(FIG / "fig2_car_paths.pdf"); fig.savefig(FIG / "fig2_car_paths.png", dpi=160); plt.close(fig)

# ---- Fig 3: per-model behaviour in the mixed arm (share 1.0) -------------
agg = {}
with open(OUT / "mixed_share1.00/llm_calls.jsonl") as f:
    for l in f:
        d = json.loads(l)
        if d["t"] < 1795544800 or d.get("parsed") is None: continue
        w = "pre\n(days 0–9)" if d["step"] < 10 else ("news\n(10–19)" if d["step"] < 20 else "post\n(20–29)")
        a = agg.setdefault((d["provider"], w), [0, 0, 0, 0.0, 0.0]); a[0] += 1
        tr = [t for t in (d["parsed"].get("trades") or []) if isinstance(t, dict)]
        s = [t for t in tr if str(t.get("action", "")).upper() == "SELL"]; b = [t for t in tr if str(t.get("action", "")).upper() == "BUY"]
        a[1] += bool(s); a[2] += bool(b); a[3] += sum(float(t.get("fraction") or 0) for t in s); a[4] += sum(float(t.get("fraction") or 0) for t in b)
wins = ["pre\n(days 0–9)", "news\n(10–19)", "post\n(20–29)"]
fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2))
for ax, (idx, ttl, ylab) in zip(axes, [(1, "share of calls that SELL", "% of calls"), (3, "requested sell size per call", "fraction of portfolio")]):
    xs = np.arange(3); w = 0.36
    for k, (prov, name, col) in enumerate([("anthropic", "Haiku 5.5", "#2a78d6"), ("openrouter_a", "DeepSeek V4.1 Flash", "#eda100")]):
        vals = []
        for win in wins:
            n, s, b, sf, bf = agg[(prov, win)]
            vals.append(100 * s / n if idx == 1 else sf / n)
        bars = ax.bar(xs + (k - 0.5) * (w + 0.02), vals, w, color=col, label=name, lw=0)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.0f}" if idx == 1 else f"{v:.3f}", ha="center", va="bottom", fontsize=7, color="#0b0b0b")
    ax.set_xticks(xs); ax.set_xticklabels(wins, fontsize=8); ax.set_title(ttl, fontsize=9); ax.set_ylabel(ylab); ax.grid(axis="x", alpha=0)
axes[0].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "fig3_mixed_behaviour.pdf"); fig.savefig(FIG / "fig3_mixed_behaviour.png", dpi=160); plt.close(fig)

# ---- Fig 4: episode-level paired effects at share 1.0 (scary) ------------
p = df[(df.share == 1.0) & (df.scary)].pivot(index="episode_id", columns="arm", values="d_car_spy_d5") * 100
fig, ax = plt.subplots(figsize=(5.5, 2.1))
for i, arm in enumerate(["mono", "persona", "mixed"]):
    y = np.full(len(p), i) + rng.uniform(-0.12, 0.12, len(p))
    ax.scatter(p[arm], y, s=18, color=COL[arm], alpha=0.75, lw=0)
    m, lo, hi = boot(p[arm].values)
    ax.plot([lo, hi], [i + 0.32, i + 0.32], color=COL[arm], lw=2); ax.plot(m, i + 0.32, "D", color=COL[arm], ms=5)
    ax.text(m, i + 0.42, f"mean {m:+.1f} pp", ha="center", fontsize=7, color="#0b0b0b")
    neg = int((p[arm] < 0).sum()); ax.text(14.5, i, f"{neg}/24 negative", va="center", fontsize=7, color="#52514e")
ax.axvline(0, color="#52514e", lw=0.8); ax.set_yticks([0, 1, 2]); ax.set_yticklabels([LABEL[a] for a in ["mono", "persona", "mixed"]], fontsize=8)
ax.set_xlabel("per-episode SPY CAR days 0–5 vs. control (pp), share 1.0, scary episodes"); ax.set_xlim(-45, 22); ax.grid(axis="y", alpha=0)
fig.tight_layout(); fig.savefig(FIG / "fig4_episode_effects.pdf"); fig.savefig(FIG / "fig4_episode_effects.png", dpi=160); plt.close(fig)
print("wrote", sorted(str(p) for p in FIG.glob("*.pdf")))
