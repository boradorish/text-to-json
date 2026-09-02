"""Re-score BFCL-v4 offline results with a JSON-native decode_ast (official ast_checker unchanged)."""
from __future__ import annotations
import argparse
import ast, json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.constants.eval_config import PROMPT_PATH, POSSIBLE_ANSWER_PATH
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_ast_entry
from bfcl_eval.model_handler.utils import default_decode_ast_prompting
import summarize_bfcl

ROOT = Path(__file__).resolve().parents[1] / "outputs" / "bfcl"
CATS = ["simple_python", "multiple", "parallel"]

def read_jsonl(p): return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]

def _tuples_to_lists(o):
    if isinstance(o, tuple): return [_tuples_to_lists(x) for x in o]
    if isinstance(o, list): return [_tuples_to_lists(x) for x in o]
    if isinstance(o, dict): return {k: _tuples_to_lists(v) for k, v in o.items()}
    return o

class MultiDict(dict):
    """dict that also remembers duplicate keys as an ordered (k, v) list."""
    def __init__(self, pairs):
        super().__init__(pairs); self.pairs = list(pairs)

def _pairs(pairs):
    return MultiDict(pairs)

NAME_KEYS = ("name", "func", "func_name", "function", "function_name", "tool", "tool_name")
ARGS_KEYS = ("arguments", "args", "params", "parameters", "input", "inputs")

def loads_lenient(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t).strip()
    try:
        return json.loads(t, object_pairs_hook=_pairs)
    except Exception:
        pass
    # python-literal fallback (handles tuples like (1.0, 2.0))
    t2 = re.sub(r"\btrue\b", "True", t); t2 = re.sub(r"\bfalse\b", "False", t2); t2 = re.sub(r"\bnull\b", "None", t2)
    return _tuples_to_lists(ast.literal_eval(t2))

def canonical_name(key: str, funcs: set[str]) -> str:
    if key in funcs: return key
    m = re.match(r"^(.*?)[\s_\-]*(?:\(\d+\)|\d+)$", key)
    if m and m.group(1) in funcs: return m.group(1)
    return key

class JsonHandler:
    """Duck-typed BFCL handler: decode STAGE-style JSON into [{func: {arg: val}}]."""
    funcs: set[str] = set()
    props: set[str] = set()
    def decode_ast(self, result, return_format=ReturnFormat.PYTHON, has_tool_call_tag=False):
        if not isinstance(result, str): raise ValueError("non-string result")
        obj = loads_lenient(result)
        calls: list[dict] = []
        def items(o):
            return o.pairs if isinstance(o, MultiDict) else list(o.items())
        def consume(o):
            if isinstance(o, dict) and o and all(isinstance(v, dict) for v in dict(items(o)).values()) and all(isinstance(v, dict) for _, v in items(o)):
                for k, v in items(o): calls.append({canonical_name(k, self.funcs): dict(v)})
            elif isinstance(o, dict) and any(k in o for k in NAME_KEYS) and isinstance(o.get(next(k for k in NAME_KEYS if k in o)), str) \
                    and (o[next(k for k in NAME_KEYS if k in o)] in self.funcs):
                nk = next(k for k in NAME_KEYS if k in o); name = o[nk]
                ak = next((k for k in ARGS_KEYS if k in o and isinstance(o[k], dict)), None)
                args = dict(o[ak]) if ak else {k: v for k, v in o.items() if k != nk}
                calls.append({name: args})
            elif isinstance(o, dict) and len(o) == 1 and isinstance(next(iter(o.values())), list) and all(isinstance(x, dict) for x in next(iter(o.values()))):
                # wrapper like {"func_call": [...]} / {"calls": [...]} / {"results": [...]}
                for x in next(iter(o.values())): consume(x)
            elif isinstance(o, dict) and len(self.funcs) == 1 and set(o) <= self.props and o:
                # flat argument dict for the single offered function (simple category only)
                calls.append({next(iter(self.funcs)): dict(o)})
            elif isinstance(o, dict) and o and all(isinstance(v, str) for v in o.values()) and all("(" in v for v in o.values()):
                for v in o.values(): calls.extend(default_decode_ast_prompting(v))
            elif isinstance(o, dict) and set(o) >= {"name", "arguments"} and isinstance(o["arguments"], dict):
                calls.append({o["name"]: o["arguments"]})
            elif isinstance(o, list):
                for x in o: consume(x)
            else:
                raise ValueError(f"unrecognized JSON shape: {type(o).__name__}")
        consume(obj)
        if not calls: raise ValueError("no calls decoded")
        return calls

handler = JsonHandler()

_orig_summarize = summarize_bfcl.summarize_category
def json_decode_for_diag(raw):
    try:
        return handler.decode_ast(raw)
    except Exception:
        return []
def summarize_with_funcs(category, result_path):
    # re-implement the per-item loop so funcs/props are set before decode
    questions, answers = summarize_bfcl.load_reference(category)
    rows = {row["id"]: row for row in summarize_bfcl.read_jsonl(result_path)}
    from collections import Counter as _C
    counters = _C()
    # Ablations may deliberately evaluate a prefix/subset.  Score only result
    # IDs present on disk rather than counting absent reference items as fails.
    for item_id, question in questions.items():
        if item_id not in rows:
            continue
        handler.funcs = {f["name"] for f in question["function"]}
        handler.props = set(question["function"][0]["parameters"]["properties"]) if len(question["function"]) == 1 else set()
        prediction = json_decode_for_diag(rows.get(item_id, {}).get("result"))
        gold_calls = answers[item_id]["ground_truth"]
        counters["examples"] += 1; counters["expected_calls"] += len(gold_calls)
        unused = list(range(len(prediction))); matched = []
        for gold in gold_calls:
            name = next(iter(gold))
            chosen = next((i for i in unused if name in prediction[i]), None)
            if chosen is None: continue
            unused.remove(chosen); counters["function_selected"] += 1
            function = next((fn for fn in question["function"] if fn["name"] == name), None)
            if function: matched.append((gold, prediction[chosen], function))
        for gold, pred, function in matched:
            name = next(iter(gold)); expected_args = gold[name]; pred_args = pred[name]
            details = function["parameters"]["properties"]; required = set(function["parameters"].get("required", []))
            schema_valid = required.issubset(pred_args) and set(pred_args).issubset(details)
            if schema_valid: schema_valid = all(summarize_bfcl.type_ok(v, details[k]) for k, v in pred_args.items())
            counters["matched_calls"] += 1; counters["argument_schema_valid"] += int(schema_valid)
            for key, allowed in expected_args.items():
                counters["expected_arguments"] += 1
                if key in pred_args and pred_args[key] in allowed: counters["argument_value_correct"] += 1
    r = lambda a, b: counters[a] / counters[b] if counters[b] else 0.0
    return {"category": category, **dict(counters), "function_selection_accuracy": r("function_selected", "expected_calls"),
            "argument_schema_validity": r("argument_schema_valid", "matched_calls"), "argument_value_accuracy": r("argument_value_correct", "expected_arguments")}
summarize_bfcl.summarize_category = summarize_with_funcs

def rescore(run: str, model_dir: str = "qwen3-4b", categories: list[str] | None = None):
    out_dir = ROOT / run / "score" / "json_decoder"; out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    categories = categories or CATS
    for cat in categories:
        prompts = {r["id"]: r for r in read_jsonl(PROMPT_PATH / f"BFCL_v4_{cat}.json")}
        answers = {r["id"]: r for r in read_jsonl(POSSIBLE_ANSWER_PATH / f"BFCL_v4_{cat}.json")}
        results = read_jsonl(ROOT / run / "result" / model_dir / "non_live" / f"BFCL_v4_{cat}_result.json")
        errors = Counter(); failed = []; correct = 0
        for r in results:
            p = prompts[r["id"]]; handler.funcs = {f["name"] for f in p["function"]}
            handler.props = set(p["function"][0]["parameters"]["properties"]) if len(p["function"]) == 1 else set()
            e = _evaluate_single_ast_entry(handler, r["id"], r["result"], answers[r["id"]]["ground_truth"], p,
                                           model_dir, cat, language=Language.PYTHON, return_format=ReturnFormat.PYTHON)
            if e["valid"]: correct += 1
            else: errors[e["error_type"]] += 1; failed.append(e)
        acc = correct / len(results)
        summary[cat] = {"accuracy": acc, "correct": correct, "total": len(results), "errors": dict(errors)}
        with (out_dir / f"BFCL_v4_{cat}_score.json").open("w") as fh:
            fh.write(json.dumps({"accuracy": acc, "correct_count": correct, "total_count": len(results)}) + "\n")
            for e in failed: fh.write(json.dumps(e, default=str) + "\n")
    # diagnostics with JSON decoder
    diag = [summarize_with_funcs(cat, ROOT / run / "result" / model_dir / "non_live" / f"BFCL_v4_{cat}_result.json") for cat in categories]
    (out_dir / "stage_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    return summary, diag

def official(run: str, model_dir: str = "qwen3-4b"):
    out = {}
    for cat in CATS:
        p = ROOT / run / "score" / model_dir / "non_live" / f"BFCL_v4_{cat}_score.json"
        out[cat] = json.loads(p.open().readline())["accuracy"]
    return out

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score BFCL output with the JSON-native decoder.")
    parser.add_argument("--runs", nargs="+", default=["qwen3_4b_sft", "qwen3_4b_base"])
    parser.add_argument("--baseline", default="qwen3_4b_base", help="Run whose official scores are shown for comparison.")
    parser.add_argument("--model-dir", default="qwen3-4b", help="BFCL result subdirectory below result/.")
    parser.add_argument("--categories", nargs="+", choices=CATS, default=CATS)
    args = parser.parse_args()
    baseline_off = None
    try:
        baseline_off = official(args.baseline, args.model_dir)
    except FileNotFoundError:
        pass
    summaries = {}
    for run in args.runs:
        summary, diagnostics = rescore(run, args.model_dir, args.categories)
        summaries[run] = (summary, diagnostics)
    print("\n=== JSON-native AST accuracy (%) ===")
    header = "category" + (f"  {args.baseline} official" if baseline_off else "")
    header += "".join(f"  {run}" for run in args.runs)
    print(header)
    for cat in args.categories:
        values = [f"{cat:14}"]
        if baseline_off:
            values.append(f"{baseline_off[cat] * 100:>18.1f}")
        values.extend(f"{summaries[run][0][cat]['accuracy'] * 100:>10.1f}" for run in args.runs)
        print(" ".join(values))
    for run, (summary, diagnostics) in summaries.items():
        print(f"\n=== {run}: function selection / schema validity / value accuracy (%) ===")
        for row in diagnostics:
            print(f"{row['category']:14} {row['function_selection_accuracy']*100:.1f} / {row['argument_schema_validity']*100:.1f} / {row['argument_value_accuracy']*100:.1f}")
        print(f"=== {run}: error breakdown ===")
        for cat in args.categories:
            print(cat, summary[cat]["errors"])


if __name__ == "__main__":
    main()
