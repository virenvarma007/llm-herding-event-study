# arXiv submission package

Files to upload: `main.tex`, `refs.bib`, `figs/fig1_dose_response.pdf`, `figs/fig2_car_paths.pdf`,
`figs/fig3_mixed_behaviour.pdf`, `figs/fig4_episode_effects.pdf` (zip them keeping the `figs/` folder).

No LaTeX is installed on this machine, so the manuscript has not been compiled here.

## Before you upload (30 minutes)

1. **Compile once on Overleaf** (free): New Project -> Upload Project -> the zip. Compiler: pdfLaTeX. Fix any error it shows.
2. In `main.tex` replace `TODO@gatech.edu` and the `TODO: repository URL` line (push `herd2/`, `config.yaml`, `requirements.txt`, `README.md` to a public GitHub repo; do NOT push `.env`, `out/llm_cache.sqlite`, or the CRSP CSVs).
3. In `refs.bib` fix the entries marked `TODO` (author lists for hansen2025, twinmarket2025, xia2026, scis2026). Open each arXiv/DOI page and copy the authors.
4. Decide the affiliation line. "Georgia Institute of Technology" is what your Scholar profile says. Adding Amazon may require Amazon's publication approval; check your employment agreement first.
5. Delete the HTML-comment reminder block at the bottom of `../PAPER_DRAFT.md` if you ever export that file; `main.tex` does not contain it.

## arXiv settings

- Primary category: **q-fin.TR** (Trading and Market Microstructure). Cross-lists: **cs.MA** (Multiagent Systems), **cs.CE**, **cs.AI**.
- License: arXiv.org perpetual non-exclusive (default) is fine for later journal submission to Springer (JEIC) and TMLR.
- First-time submitters in q-fin may need an endorsement; a gatech.edu email usually auto-qualifies. If asked, the endorsement request form names people who can endorse.
- Submissions announced at 20:00 US Eastern on weekdays; submit before 14:00 ET to be announced the same evening.

## After it is live

- Add the arXiv ID to your Google Scholar profile (it appears automatically within days; if not, "Add article manually").
- Submit the same manuscript to the JEIC special issue "Thirty Years of WEHIA" (deadline 15 Dec 2026) — Springer allows arXiv preprints. Add a two-paragraph finance-oriented opening to the introduction for that version.

## Regenerating figures

```
cd ..
.venv/bin/python -m herd2.paper_figs      # writes paper/figs/*.pdf and *.png
```
