# Paper figures profile (NeurIPS 2026 SLM-Agents workshop version)

Project-owned parameters for `paper-figure-plot` / `paper-writing`. When a skill
default and this file disagree, this file wins.

| Field | Value |
|---|---|
| Venue & column model | NeurIPS 2026 workshop, `neurips_2026.sty`, single column, `\textwidth` = 5.5 in. Max asset width 5.5 in. |
| Preamble facts | `graphicx`, `booktabs`, `threeparttable`, `siunitx`, `xcolor[table]`, `hyperref` loaded. `subcaption` added for the workshop appendix figures. |
| Palette | STAGE-trained arms: `#1E6E8A` (teal-blue); base / untrained arms: `#8A6A1A` (ochre); neutral grey `#6B7580`. Redundant encoding: marker shape = initialisation (circle = base init, square = STAGE-SFT init), fill = with STAGE-Dialog continuation. |
| Typography | Times (matches `figure1.pdf`, TimesNewRomanPSMT). matplotlib: `font.family=serif`, `font.serif=[Times New Roman, Times, STIXGeneral]`, `mathtext.fontset=stix`. 7 pt base, 6.2 pt ticks, 5.6 pt in-plot notes. |
| Renderer inventory | `benchmark/paper_figures/fig_length.py` reads `data/length_buckets/{realkie_header,extractbench_long}.json`, writes `fig_len_a.pdf`, `fig_len_b.pdf`. `benchmark/paper_figures/fig_tables_to_plots.py` writes `fig_main_*`, `fig_dje_*`, `fig_eb_*` from `paper_data.json` and the manuscript tables. `benchmark/paper_figures/fig_sgd.py` reads `benchmark/paper_figures/data/paper_data.json`, writes `overleaf-paper/figures/fig_sgd_a.pdf`, `fig_sgd_b.pdf` (2.70 in x 1.95 in each). `benchmark/paper_figures/emit_tables.py` writes `overleaf-paper/tables/*.tex` (incl. `tab_realkie`, `tab_realkie_spans`, `tab_realworld_short`) from the same data file. Run both with `venv/bin/python <script>` from the repo root. |
| Figure-script status | Both scripts are frozen one-offs for the workshop submission; edit in place and re-run, never hand-edit the emitted assets. |
| Data provenance | `paper_data.json` is built on the pod from raw `outputs/*.jsonl` with `benchmark/evaluate.py` (STAGE-Eval, ExtractBench) and `benchmark/evaluate_sgd.py` (SGD official metric); see `_provenance` inside the file. |
| Notation | `\method` = STAGE, `\bench` = STAGE-Eval. Metrics: PFR, EMR, SCR, NR, VA (paper), JGA (SGD). In figures "PFR" is the paper's failure-rate definition unless the axis says "parse success". |
| Naming | "Qwen3-4B" (base, thinking off), "Qwen3-4B (thinking)" for the thinking-on control, "+ STAGE" for the STAGE SFT model, "+ STAGE + STAGE-Dialog" for the continuation. |
| Paper checkout | `overleaf-paper/` (Overleaf git clone, gitignored from the main repo). Commit locally; push only on the author's explicit request via `overleaf-git` skill. |
| Edit marking | All workshop-appendix prose written by the agent is wrapped in `\CLAUDE{...}` (violet); tables emitted by scripts start with `\CLAUDEcolor`. The older `\claude{}` (blue) marks the earlier main-text edits. |
| Local overrides | Appendix figure uses two side-by-side scatter panels at 2.70 x 1.95 in (assembly ~2.8:1) instead of the 7:3 single-plot default because the finding is a two-axis trade-off. |
