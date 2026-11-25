# AI Herd v2 -- event-study design

Does a higher share of LLM-driven traders make a market react more to a scary rumor, and does
**model diversity** (several LLMs) break the herd where **persona diversity** (one LLM, several
prompts) does not?

Paper: `paper/main.tex` (arXiv preprint, November 2026). Results tables: `out/*.csv`, `out/regression.txt`.
Per-episode outputs (`metrics.json`, `timeseries.csv`) for all 10 configurations are included; raw LLM call
logs (~600 MB) and the CRSP price CSVs (licensed) are not. Put your own daily price CSVs with columns
`date,ticker,close,adj_close` in a folder and point `data_dir` in `config.yaml` at it to re-run.

## Headline results (36 episodes, 100 agents, SPY abnormal return days 0-5 vs share-0 control)

| Arm | share 0.25 | share 0.5 | share 1.0 | placebo (any share) |
|---|---|---|---|---|
| mono (Haiku 5.5) | -1.2 pp | -1.5 pp | -5.2 pp | ~0 |
| persona (Haiku, 5 prompts) | -0.7 pp | -0.8 pp | -2.6 pp | ~0 |
| mixed (Haiku + DeepSeek) | -1.4 pp | -2.0 pp | -17.4 pp | +2 to +4 pp at low share |

share x scary interaction -0.086 (p < 0.001, SE clustered by episode); x mixed -0.119 (p < 0.001); x persona +0.009 (n.s.).

## Design

| | |
|---|---|
| Episodes | 36 windows of 30 trading days, sampled once with a fixed seed from 2016-2023, identical for every config (paired design) |
| Shock | day 10: scary rumor (24 episodes) or neutral placebo news (12 episodes), shown to LLM agents for 10 days |
| Agents | 100: rule traders (value / trend / noise, heterogeneous parameters, trade size proportional to wealth) + LLM traders |
| LLM share | 0 (control), 0.25, 0.5, 1.0 -- the same random slots are converted to LLM agents, so configs differ only in who is LLM-driven |
| Arms | `mono` = Claude Haiku 5.5, one prompt · `persona` = Haiku, 5 personas · `mixed` = 50% Haiku + 50% DeepSeek V4.1 Flash (OpenRouter), one prompt |
| Market | real returns drive the fundamental; agents add price impact `alpha * net_shares / depth`; depth = 5x the population's holdings |
| Outcome | **abnormal return** = simulated minus fundamental (history cancels out), CAR days 0-2/5/10 after the shock |
| Also | unique-agent sell fraction (LLM and rule), spread/depth change, cross-asset correlation of abnormal returns, parse-failure rate |

Primary test: `CAR_SPY_d5 ~ share * scary * arm`, SE clustered by episode; plus paired episode-level
effects vs share 0 with bootstrap CIs.

## Setup

```bash
cd "v2_event_study"
python3 -m venv .venv && source .venv/bin/activate     # a .venv already exists on the original machine
pip install -r requirements.txt
cp .env.example .env     # put ANTHROPIC_API_KEY and OPENROUTER_API_KEY in it
```

## Measured on 2026-11-23 (1 real episode, 25 Haiku agents, 750 calls)

| | |
|---|---|
| Haiku 5.5 | 644 in / ~100 out tokens per call, $0.000126 per call, 0 parse failures |
| DeepSeek V4.1 Flash | ~500 in / ~130 out, $0.00013 per call |
| Full sweep (517.5k valid calls) | **~$52 Claude + ~$10 OpenRouter**, ~8 h with 10 runs in parallel |
| Behaviour | Haiku sold on the rumor day (72% of LLM agents vs ~25% on normal days) and stopped buying for 3 days |
| Prompt sensitivity | explicit per-asset HOLD wording -> 0% sold ("await confirmation"). Kept as `prompt_variant: per_asset` for a robustness check |
| Data | prices rescaled to 100 at episode start; split-adjusted CRSP levels made Haiku distrust the data and hint at the date |

## Run

```bash
python -m herd2.run plan                                   # calls + cost estimate
python -m herd2.run run --arm mono --share 0.5 --provider mock --episodes 2   # pipeline test, no API calls
python -m herd2.run check-models                           # 1 real call per provider; verifies ids, keys, prices
python -m herd2.run run --arm mono --share 0.25 --episodes 1                 # one real episode (~750 calls, cents)
./run_all.sh                                               # full sweep, all configs in parallel, resumable
python -m herd2.analyze                                    # tables, CIs, regression, figures -> out/
```

Hard spend caps live in `config.yaml` (`llm.spend_cap_usd`) and are enforced across all parallel
runs through `out/spend_ledger.jsonl`. Also set a cap on the Claude Console key itself.

## Outputs

```
out/<arm>_share<share>/episode_NN/{timeseries.csv, actions.csv, metrics.json}
out/<arm>_share<share>/{llm_calls.jsonl, llm_stats.json, config_resolved.json}
out/llm_cache.sqlite  out/spend_ledger.jsonl
out/{all_episodes.csv, summary_by_config.csv, paired_effects_vs_share0.csv, regression.txt,
     fig_dose_response.png, fig_car_paths.png}
```

## Fixes relative to v1

- model that can read news (Haiku 5.5 / strong OpenRouter models), JSON schema output, 600-token budget
- agents start 50% invested, so SELL is possible
- rule agents no longer identical clones; sizes scale with wealth
- sell synchronicity = unique agents / agents, capped at 1
- outcome is abnormal return, so the 2018/2020 history cannot leak in
- placebo news arm, share-0 control, paired episodes, bootstrap CIs, clustered regression

## Citation

```
@article{varma2026herd,
  title  = {Model Diversity Does Not Break the Herd: Prompt vs. Model Diversity in LLM Agent Populations under Shared Information Shocks},
  author = {Varma, Viren},
  journal= {arXiv preprint},
  year   = {2026}
}
```

MIT License. See `LICENSE`.
