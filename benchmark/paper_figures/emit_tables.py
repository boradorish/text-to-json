"""Emit the workshop-appendix tables as .tex from benchmark/paper_figures/data/paper_data.json.

Numbers in the manuscript tables therefore come from the same data file as the
figures. Each table starts with \\CLAUDEcolor so the agent-written content is
visible in the compiled PDF.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = json.loads((ROOT / "benchmark" / "paper_figures" / "data" / "paper_data.json").read_text())
OUT = ROOT / "overleaf-paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def f1(x: float) -> str:
    return f"{x:.1f}"


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")
    print("wrote", OUT / name)


# ---------------------------------------------------------------- Table: constrained decoding
se = D["stage_eval"]
rows = [
    ("Qwen3-4B", "free", "base_nothink_free"),
    ("Qwen3-4B", "xgrammar", "base_nothink_xgrammar"),
    ("Qwen3-4B + \\method", "free", "stage_sft_free"),
    ("Qwen3-4B + \\method", "xgrammar", "stage_sft_xgrammar"),
]
lines = []
for model, dec, key in rows:
    m = se[key]["compat798"]
    bold = key == "stage_sft_free"
    cells = [f1(100 - m["PFR"]), f1(m["EMR"]), f1(m["SCR"]), f1(m["NR"]), f1(m["VA"])]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    lines.append(f"{model} & {dec} & " + " & ".join(cells) + " \\\\")
tab_cd = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Decoding} & PFR$\downarrow$ & EMR$\uparrow$ & SCR$\uparrow$ & NR$\downarrow$ & VA$\uparrow$ \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:]) + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{Grammar-constrained decoding raises structure but not values; \method{} training raises both.} \bench{} restricted to the 798 of 851 test schemas that xgrammar compiles under vLLM, so all four rows share one denominator. Qwen3-4B is run with thinking disabled. xgrammar constrains generation with each example's own JSON Schema. PFR is the parse-failure rate.}}
\label{tab:app_constrained}
\end{table}
"""
write("tab_constrained_decoding.tex", tab_cd)

# ---------------------------------------------------------------- Table: inference cost
cost = {(r["label"], r["pass"], r["batch_size"]): r for r in D["inference_cost"]}


def c(label, pas, bs, key, scale=1.0, nd=2):
    return f"{float(cost[(label, pas, str(bs))][key]) * scale:.{nd}f}"


cost_rows = [
    ("Qwen3-4B", "free", "base_nothink_free"),
    ("Qwen3-4B", "xgrammar", "base_nothink_xgrammar"),
    ("Qwen3-4B + \\method", "free", "sft_free"),
    ("Qwen3-4B + \\method", "xgrammar", "sft_xgrammar"),
]
lines = []
for model, dec, lab in cost_rows:
    compile_ms = "--" if "xgrammar" not in lab else c(lab, "warm", 1, "grammar_compile_median_seconds", 1000, 1)
    lines.append(
        f"{model} & {dec} & {c(lab,'warm',1,'latency_median_seconds')} & {c(lab,'warm',1,'latency_p90_seconds')} & "
        f"{c(lab,'throughput',32,'examples_per_second')} & {c(lab,'warm',1,'mean_generated_tokens',1,0)} & {compile_ms} \\\\"
    )
tab_cost = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
\textbf{Model} & \textbf{Decoding} & \shortstack{Latency\\median (s)} & \shortstack{Latency\\p90 (s)} & \shortstack{Throughput\\(ex/s, batch 32)} & \shortstack{Generated\\tokens} & \shortstack{Grammar\\compile (ms)} \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:]) + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{\method{} training adds no inference-time cost; grammar constraints add about 10\% latency and 15\% throughput.} Same 798 examples and one H200 as Table~\ref{tab:app_constrained}; batch-1 latency is the warm-cache pass, grammar compile time is measured per schema with the xgrammar cache disabled. Per-token time is 4.5\,ms in every row, so latency differences follow generated length.}}
\label{tab:app_cost}
\end{table}
"""
write("tab_inference_cost.tex", tab_cost)

# ---------------------------------------------------------------- Table: ExtractBench
eb = D["extractbench_194"]
eb_rows = [
    ("Qwen3-4B", "free", "qwen3_base_nothink_free"),
    ("Qwen3-4B", "xgrammar", "qwen3_base_nothink_xgrammar"),
    ("Qwen3-4B + \\method", "free", "qwen3_sft_free"),
    ("Qwen3-4B + \\method", "xgrammar", "qwen3_sft_xgrammar"),
    ("Qwen2.5-3B-Instruct", "free", "qwen25_base_free"),
    ("Qwen2.5-3B-Instruct", "xgrammar", "qwen25_base_xgrammar"),
    ("Qwen2.5-3B-Instruct + \\method", "free", "qwen25_sft_free"),
    ("Qwen2.5-3B-Instruct + \\method", "xgrammar", "qwen25_sft_xgrammar"),
]
lines = []
for model, dec, key in eb_rows:
    a, s, m = eb[key]["all194"], eb[key]["short"], eb[key]["medium"]
    lines.append(f"{model} & {dec} & {f1(100 - a['PFR'])} & {f1(a['SCR'])} & {f1(a['VA'])} & {f1(s['VA'])} & {f1(m['VA'])} & {f1(100 - m['PFR'])} \\\\")
tab_eb = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}
\toprule
 & & \multicolumn{3}{c}{All (194)} & Short (137) & \multicolumn{2}{c}{Medium, $>$8k tokens (57)} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-6} \cmidrule(lr){7-8}
\textbf{Model} & \textbf{Decoding} & PFR$\downarrow$ & SCR$\uparrow$ & VA$\uparrow$ & VA$\uparrow$ & VA$\uparrow$ & PFR$\downarrow$ \\
\midrule
""" + "\n".join(lines[:4]) + r"""
\midrule
""" + "\n".join(lines[4:]) + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{On real documents, \method{} training keeps outputs parseable and helps most on documents longer than the 8k-token training window.} ExtractBench digital-text documents that fit a 32,768-token context (200) minus 6 schemas xgrammar cannot compile, so every row scores the same 194 documents; Short and Medium are ExtractBench's own length splits. Qwen3-4B runs with thinking disabled. Exact-match is zero for every row and omitted.}}
\label{tab:app_extractbench}
\end{table}
"""
write("tab_extractbench.tex", tab_eb)

# ---------------------------------------------------------------- Table: SGD dialogue state tracking
sgd = D["sgd_full_2000"]
sgd_rows = [
    ("Qwen3-4B (thinking)", "qwen3_4b_base", "base_think_free"),
    ("Qwen3-4B", "qwen3_4b_base_nothink", "base_nothink_free"),
    ("Qwen3-4B + \\method", "qwen3_4b_sft", "stage_sft_free"),
    ("Qwen3-4B + \\method{} + \\method-Dialog", "qwen3_4b_stage_dialog_v2", "stage_sft_dialog_v2"),
    ("Qwen3-4B + \\method-Dialog", "qwen3_4b_base_dialog_v2", "base_dialog_v2"),
]
lines = []
for model, key, sekey in sgd_rows:
    st, ex = sgd[f"{key}_standard_full"]["all"], sgd[f"{key}_explicit_full"]["all"]
    va = se[sekey]["all"]["VA"]
    cells = [f1(st["joint_goal_accuracy"] * 100), f1(st["hallucinated_slot_rate"] * 100), f1(st["missing_slot_rate"] * 100),
             f1(ex["joint_goal_accuracy"] * 100), f1(ex["hallucinated_slot_rate"] * 100), f1(ex["missing_slot_rate"] * 100), f1(va)]
    lines.append(f"{model} & " + " & ".join(cells) + " \\\\")
tab_sgd = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lccccccc}
\toprule
 & \multicolumn{3}{c}{SGD, specified slots only} & \multicolumn{3}{c}{SGD, every slot (\texttt{"no output"})} & \bench \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-8}
\textbf{Model} & JGA$\uparrow$ & Halluc.$\downarrow$ & Missing$\downarrow$ & JGA$\uparrow$ & Halluc.$\downarrow$ & Missing$\downarrow$ & VA$\uparrow$ \\
\midrule
""" + "\n".join(lines[:3]) + r"""
\midrule
""" + "\n".join(lines[3:]) + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{Continuing \method{} training on \method-Dialog turns a failing state tracker into the best one while keeping in-distribution extraction.} SGD test, 2,000 user turns balanced over seen and unseen services, official joint goal accuracy (JGA) with the two schema formats described in the text; Halluc.\ is the share of unspecified slots given a value and Missing the share of specified slots left empty. \bench{} VA is on all 851 test examples. No SGD data is used in training. The last row trains the same adapter from the untrained Qwen3-4B.}}
\label{tab:app_sgd}
\end{table}
"""
write("tab_sgd.tex", tab_sgd)

# ---------------------------------------------------------------- Table: STAGE-Dialog training setting
st = D["stage_dialog_stats"]
tab_train = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\small
\begin{tabular}{ll}
\toprule
Setting & Value \\
\midrule
Source records (spreadsheet rows) & 6,000 \\
Generated dialogues / passing validation & """ + f"{st['jobs']:,} / {st['valid']:,} ({st['valid']/st['jobs']*100:.1f}\\%)" + r""" \\
Dialogue generator & Qwen3-4B-Instruct-2507, temperature 0.8 \\
State examples produced / used & """ + f"{st['examples']:,} / 8,000" + r""" \\
Original \method{} examples mixed in & 2,000 \\
Adapter & LoRA, rank 16, $\alpha=32$, all linear layers \\
Epochs / learning rate & 2 / $1.0\times10^{-4}$ \\
Effective batch size / max length & 16 / 6,144 tokens \\
\bottomrule
\end{tabular}
\caption{\CLAUDE{\method-Dialog generation and continuation-training setting. The same setting trained from the untrained Qwen3-4B gives the last row of Table~\ref{tab:app_sgd}.}}
\label{tab:app_dialog_setting}
\end{table}
"""
write("tab_dialog_setting.tex", tab_train)

# ---------------------------------------------------------------- Table (main text): accuracy + cost in one compact table
main_rows = [
    ("Qwen3-4B", "free", "base_nothink_free", "base_nothink_free"),
    ("Qwen3-4B", "xgrammar", "base_nothink_xgrammar", "base_nothink_xgrammar"),
    ("Qwen3-4B + \\method", "free", "stage_sft_free", "sft_free"),
    ("Qwen3-4B + \\method", "xgrammar", "stage_sft_xgrammar", "sft_xgrammar"),
]
lines = []
for model, dec, sekey, ckey in main_rows:
    m = se[sekey]["compat798"]
    cells = [f1(100 - m["PFR"]), f1(m["EMR"]), f1(m["SCR"]), f1(m["VA"]), c(ckey, "warm", 1, "latency_median_seconds"), c(ckey, "throughput", 32, "examples_per_second")]
    if sekey == "stage_sft_free":
        cells = [f"\\textbf{{{x}}}" for x in cells]
    lines.append(f"{model} & {dec} & " + " & ".join(cells) + " \\\\")
tab_main = r"""
\begin{table}[t]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccc|cc}
\toprule
 & & \multicolumn{4}{c|}{\bench{} (798 schemas)} & \multicolumn{2}{c}{Cost (1 H200)} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-8}
\textbf{Model} & \textbf{Decoding} & PFR$\downarrow$ & EMR$\uparrow$ & SCR$\uparrow$ & VA$\uparrow$ & \shortstack{Latency (s)\\batch 1} & \shortstack{Throughput\\ex/s, batch 32} \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:]) + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{Grammar-constrained decoding fixes structure but not values; \method{} training fixes both at no inference-time cost.} \bench{} restricted to the 798 of 851 schemas that vLLM's xgrammar backend compiles, thinking disabled for the untrained model; latency is the warm-cache median. Full metrics, cold-cache latency, and grammar compile time are in Appendix~\ref{app:constrained}.}}
\label{tab:main_constrained}
\end{table}
"""
write("tab_main_constrained.tex", tab_main)

# ---------------------------------------------------------------- Table (appendix): RealKIE-FCC invoices
rk = D["realkie_fcc_74"]
rk_rows = [
    ("Qwen3-4B", "free", "qwen3_4b_base_nothink"),
    ("Qwen3-4B", "xgrammar", "qwen3_4b_base_nothink_xgrammar"),
    ("Qwen3-4B + \\method", "free", "qwen3_4b_stage_sft"),
    ("Qwen3-4B + \\method", "xgrammar", "qwen3_4b_stage_sft_xgrammar"),
    ("Qwen3-4B + \\method{} + \\method-Dialog", "free", "qwen3_4b_stage_dialog_v2"),
]
lines = []
for model, dec, key in rk_rows:
    m = rk[key]
    lines.append(f"{model} & {dec} & {f1(m['header_va'])} & {f1(m['item_field_va'])} & {f1(m['count_ok'])} & {f1(m['SCR'])} \\\\")
tab_rk = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}
\toprule
\textbf{Model} & \textbf{Decoding} & \shortstack{Header\\field acc.$\uparrow$} & \shortstack{Line-item\\field acc.$\uparrow$} & \shortstack{Item count\\correct$\uparrow$} & SCR$\uparrow$ \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:4]) + r"""
\midrule
""" + lines[4] + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{On multi-page invoices, \method{} training raises header-field and line-item accuracy over the untrained model.} RealKIE-FCC-Verified, the 74 of 75 invoices whose OCR text fits a 40,960-token context (median 4.8k tokens, 16 above 8k); one shared schema with six header fields and a \texttt{LineItems} array (up to 25 items). Header and line-item field accuracy compare values after number and whitespace normalisation, with predicted line items matched to gold items by field overlap; item count is the share of documents whose predicted list has the gold length; SCR is JSON Schema validity of the whole output. Qwen3-4B runs with thinking disabled. Exact match is zero for every row and omitted.}}
\label{tab:app_realkie}
\end{table}
"""
write("tab_realkie.tex", tab_rk)

# ---------------------------------------------------------------- Table (appendix): raw RealKIE span extraction (charities, NDA)
sp = D["realkie_spans"]
sp_rows = [
    ("Qwen3-4B", "free", "base"),
    ("Qwen3-4B", "xgrammar", "base_xgr"),
    ("Qwen3-4B + \\method", "free", "sft"),
    ("Qwen3-4B + \\method", "xgrammar", "sft_xgr"),
    ("Qwen3-4B + \\method{} + \\method-Dialog", "free", "dialog"),
]
lines = []
for model, dec, key in sp_rows:
    c, n = sp["charities_v2"][key], sp["nda"][key]
    assert c["n"] == 108 and n["n"] == 98, (c["n"], n["n"])
    lines.append(f"{model} & {dec} & {f1(c['span_recall'])} & {f1(c['span_precision'])} & {f1(c['span_F1'])} & {f1(c['halluc_fill'])} & {f1(n['span_recall'])} & {f1(n['span_precision'])} & {f1(n['span_F1'])} & {f1(n['halluc_fill'])} \\\\")
tab_sp = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{llcccccccc}
\toprule
& & \multicolumn{4}{c}{\textbf{Charity reports} (108 docs, 28 fields)} & \multicolumn{4}{c}{\textbf{NDAs} (98 docs, 3 fields)} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}
\textbf{Model} & \textbf{Decoding} & Recall$\uparrow$ & Prec.$\uparrow$ & F1$\uparrow$ & Fill$\downarrow$ & Recall$\uparrow$ & Prec.$\uparrow$ & F1$\uparrow$ & Fill$\downarrow$ \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:4]) + r"""
\midrule
""" + lines[4] + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{On real-world span extraction with verbatim gold, \method{} training keeps recall and the \method-Dialog continuation restores precision.} RealKIE charity annual reports (28 fields, median 6.7k tokens, up to 34k) and non-disclosure agreements (3 fields, median 3.6k tokens), test splits, every field an array of the document's verbatim spans (empty when absent). A predicted span matches a gold span of the same field when their normalized token sets have Jaccard $\geq 0.5$; recall and precision are micro-averaged over spans; Fill is the share of empty gold fields given a value. Qwen3-4B runs with thinking disabled.}}
\label{tab:app_realkie_spans}
\end{table}
"""
write("tab_realkie_spans.tex", tab_sp)

# ---------------------------------------------------------------- Table (appendix): short real-world verbatim sets (SWDE, FDA, PMC)
rs = D["realworld_short"]
rs_rows = [
    ("Qwen3-4B", "free", "base"),
    ("Qwen3-4B", "xgrammar", "base_xgr"),
    ("Qwen3-4B + \\method", "free", "sft"),
    ("Qwen3-4B + \\method", "xgrammar", "sft_xgr"),
    ("Qwen3-4B + \\method{} + \\method-Dialog", "free", "dialog"),
]
lines = []
for model, dec, key in rs_rows:
    m = rs[key]
    assert m["swde_n"] == 1111 and m["fda_n"] == 1102 and m["pmc_n"] == 119, (m["swde_n"], m["fda_n"], m["pmc_n"])
    lines.append(f"{model} & {dec} & {f1(m['swde_va_norm'])} & {f1(m['swde_va_strict'])} & {f1(m['fda_va_norm'])} & {f1(m['fda_va_strict'])} & {f1(m['pmc_scalar_va'])} & {f1(m['pmc_keyword_recall'])} & {f1(m['pmc_ref_record_recall'])} & {f1(m['pmc_author_field_recall'])} \\\\")
tab_rs = r"""
\begin{table}[h]
\CLAUDEcolor
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccccc}
\toprule
& & \multicolumn{2}{c}{\textbf{SWDE} (1,111)} & \multicolumn{2}{c}{\textbf{FDA 510(k)} (1,102)} & \multicolumn{4}{c}{\textbf{PMC articles} (119)} \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-10}
\textbf{Model} & \textbf{Decoding} & norm. & exact & norm. & exact & scalar & keywords & references & authors \\
\midrule
""" + "\n".join(lines[:2]) + r"""
\midrule
""" + "\n".join(lines[2:4]) + r"""
\midrule
""" + lines[4] + r"""
\bottomrule
\end{tabular}
\caption{\CLAUDE{\textbf{On short real-world documents with verbatim gold, \method{} training is neutral to slightly positive on values and the \method-Dialog continuation is the best free-decoding row on two of three sets.} SWDE web pages (one attribute per page) and FDA 510(k) decision-summary excerpts (BASED benchmark, 2k-token windows): value accuracy after case and punctuation normalisation (norm.) and exact leaf match (exact); the exact gap for the \method{} rows comes from dropped commas and final periods, not from wrong values. PMC open-access articles (median 9k tokens): accuracy on title, journal and year (scalar), keyword recall, reference-record recall (first author, year, title; 46 references per article) and author-field recall (surname, given names, affiliations), with records aligned greedily. The \method{} model loses on author fields because it splits affiliation strings into department and university parts. Qwen3-4B runs with thinking disabled.}}
\label{tab:app_realworld_short}
\end{table}
"""
write("tab_realworld_short.tex", tab_rs)
