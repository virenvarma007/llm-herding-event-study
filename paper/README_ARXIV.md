# arXiv submission package

Files to upload: `main.tex`, `refs.bib`, `figs/fig1_dose_response.pdf`, `figs/fig2_car_paths.pdf`,
`figs/fig3_mixed_behaviour.pdf`, `figs/fig4_episode_effects.pdf` (zip them keeping the `figs/` folder).

Compiled locally with `tectonic -X compile main.tex` (9 pages, no errors); `main.pdf` is the result.

## Before you upload

1. Read `main.pdf` once end to end.
2. Code is public at https://github.com/virenvarma007/llm-herding-event-study (never push `.env`, `out/llm_cache.sqlite`, or the CRSP CSVs).
3. Reference author lists were checked against arXiv/publisher pages on 2026-11-25.
4. Affiliation line is "Independent Researcher, Seattle, WA"; no email is printed.
5. Delete the HTML-comment reminder block at the bottom of `../PAPER_DRAFT.md` if you ever export that file; `main.tex` does not contain it.

## arXiv settings

- Primary category: **q-fin.TR** (Trading and Market Microstructure). Cross-lists: **cs.MA** (Multiagent Systems), **cs.CE**, **cs.AI**.
- License: arXiv.org perpetual non-exclusive (default) is fine for later journal submission to Springer (JEIC) and TMLR.
- First-time submitters in q-fin may need an endorsement; if asked, the endorsement request form names people who can endorse.
- Submissions announced at 20:00 US Eastern on weekdays; submit before 14:00 ET to be announced the same evening.

## After it is live

- Add the arXiv ID to your Google Scholar profile (it appears automatically within days; if not, "Add article manually").
- Submit the same manuscript to the JEIC special issue "Thirty Years of WEHIA" (deadline 15 Dec 2026) — Springer allows arXiv preprints. Add a two-paragraph finance-oriented opening to the introduction for that version.

## Regenerating figures

```
cd ..
.venv/bin/python -m herd2.paper_figs      # writes paper/figs/*.pdf and *.png
```
