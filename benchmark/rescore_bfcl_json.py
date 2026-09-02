"""Re-score BFCL-v4 offline results with a JSON-native decode_ast (official ast_checker unchanged)."""
from __future__ import annotations
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
    for item_id, question in questions.items():
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

def rescore(run: str):
    out_dir = ROOT / run / "score" / "json_decoder"; out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cat in CATS:
        prompts = {r["id"]: r for r in read_jsonl(PROMPT_PATH / f"BFCL_v4_{cat}.json")}
        answers = {r["id"]: r for r in read_jsonl(POSSIBLE_ANSWER_PATH / f"BFCL_v4_{cat}.json")}
        results = read_jsonl(ROOT / run / "result" / "qwen3-4b" / "non_live" / f"BFCL_v4_{cat}_result.json")
        errors = Counter(); failed = []; correct = 0
        for r in results:
            p = prompts[r["id"]]; handler.funcs = {f["name"] for f in p["function"]}
            handler.props = set(p["function"][0]["parameters"]["properties"]) if len(p["function"]) == 1 else set()
            e = _evaluate_single_ast_entry(handler, r["id"], r["result"], answers[r["id"]]["ground_truth"], p,
                                           "qwen3-4b", cat, language=Language.PYTHON, return_format=ReturnFormat.PYTHON)
            if e["valid"]: correct += 1
            else: errors[e["error_type"]] += 1; failed.append(e)
        acc = correct / len(results)
        summary[cat] = {"accuracy": acc, "correct": correct, "total": len(results), "errors": dict(errors)}
        with (out_dir / f"BFCL_v4_{cat}_score.json").open("w") as fh:
            fh.write(json.dumps({"accuracy": acc, "correct_count": correct, "total_count": len(results)}) + "\n")
            for e in failed: fh.write(json.dumps(e, default=str) + "\n")
    # diagnostics with JSON decoder
    diag = [summarize_with_funcs(cat, ROOT / run / "result" / "qwen3-4b" / "non_live" / f"BFCL_v4_{cat}_result.json") for cat in CATS]
    (out_dir / "stage_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    return summary, diag

def official(run: str):
    out = {}
    for cat in CATS:
        p = ROOT / run / "score" / "qwen3-4b" / "non_live" / f"BFCL_v4_{cat}_score.json"
        out[cat] = json.loads(p.open().readline())["accuracy"]
    return out

base_off = official("qwen3_4b_base"); sft_off = official("qwen3_4b_sft")
sft_sum, sft_diag = rescore("qwen3_4b_sft")
base_sum, base_diag = rescore("qwen3_4b_base")  # sanity: JSON decoder on base (should mostly fail -> it outputs python calls)
base_diag_off = json.load((ROOT / "qwen3_4b_base/score/stage_diagnostics.json").open())

print("\n=== AST accuracy (%) ===")
print(f"{'cat':14}{'base official':>15}{'sft official':>14}{'sft JSON-decoder':>18}")
for cat in CATS:
    print(f"{cat:14}{base_off[cat]*100:15.1f}{sft_off[cat]*100:14.1f}{sft_sum[cat]['accuracy']*100:18.1f}")
print("\n=== SFT error breakdown under JSON decoder ===")
for cat in CATS: print(cat, sft_sum[cat]["errors"])
print("\n=== Diagnostics (%): function selection / arg schema validity / arg value accuracy ===")
for b, s in zip(base_diag_off, sft_diag):
    print(f"{b['category']:14} base {b['function_selection_accuracy']*100:5.1f} / {b['argument_schema_validity']*100:5.1f} / {b['argument_value_accuracy']*100:5.1f}"
          f"   |  sft(JSON) {s['function_selection_accuracy']*100:5.1f} / {s['argument_schema_validity']*100:5.1f} / {s['argument_value_accuracy']*100:5.1f}")
print("\n=== sanity: base under JSON decoder (expect ~0) ===")
for cat in CATS: print(cat, round(base_sum[cat]["accuracy"]*100, 1))
