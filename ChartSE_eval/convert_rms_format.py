# Script to convert the format of the ground truth and predictions to the same format as TinyChart
# such that the TinyChart code for evaluation (using RMS metric) can be used.

import json

# chartqa_test_data = json.load(open("../mPLUG-DocOwl/TinyChart/data/test.json"))
# chartqa_test_data = json.load(open("data/ChartQA_test_human_filter.json"))
chartqa_test_data = json.load(open("data/epicurves_gt.json"))
# model_predictions = json.load(open("data/example_pred(use_this).json"))
model_predictions = json.load(open("data/hard_epicurves_preds.json"))

import json
import re
from typing import Optional, Tuple


def _strip_to_json_start(s: str) -> str:
    """Keep from the first '{' or '[' onward (common if there was a prefix)."""
    m = re.search(r"[\{\[]", s)
    return s[m.start() :] if m else s


def _balance_quotes_and_brackets(s: str) -> str:
    """Close open quotes/brackets outside strings; drop trailing comma outside strings."""
    s = s.rstrip()

    # If it ends with a lone backslash, duplicate it so it's a valid escape
    if s.endswith("\\"):
        s += "\\"

    out = []
    stack = []  # holds expected closers: '}' or ']'
    in_str = False
    escape = False
    # Track quote parity and count braces/brackets outside strings
    for ch in s:
        out.append(ch)
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch == "}" or ch == "]":
                if stack and stack[-1] == ch:
                    stack.pop()
                else:
                    # stray closer; ignore mismatch (keep char)
                    pass

    # Remove a trailing comma that sits outside a string
    # (e.g., '{"a":"b",   ' or after a value)
    # Walk backwards ignoring whitespace
    i = len(out) - 1
    while i >= 0 and out[i].isspace():
        i -= 1
    if i >= 0 and out[i] == ",":
        # Ensure that comma is outside strings
        # (We can re-scan a small suffix to confirm.)
        out = out[:i]  # drop the comma and any trailing ws (we re-append ws below)

    s2 = "".join(out).rstrip()

    # If we are inside a string (odd quote parity), close it.
    # Recompute in_str state for the current buffer.
    in_str = False
    escape = False
    for ch in s2:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
    if in_str:
        s2 += '"'

    # Recompute bracket stack outside strings, then append missing closers.
    stack = []
    in_str = False
    escape = False
    for ch in s2:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch in ("]", "}"):
                if stack and stack[-1] == ch:
                    stack.pop()
                else:
                    # ignore mismatched closer
                    pass

    s2 += "".join(reversed(stack))  # close in LIFO order
    return s2


def _simple_repair(s: str) -> str:
    """Minimal, fast repairs."""
    s = _strip_to_json_start(s)
    s = _balance_quotes_and_brackets(s)
    return s


def safe_json_loads_mending(s: str, *, max_trim: int = 2000) -> Optional[object]:
    """
    Try to parse JSON; if it fails, attempt to mend truncated input by
    closing strings/brackets and trimming the tail progressively.
    """
    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # First try simple repair on the whole string
    fixed = _simple_repair(s)
    try:
        return json.loads(fixed)
    except Exception:
        pass

    # Progressive backoff: trim from the end and repair, keeping as much as possible
    s_clean = _strip_to_json_start(s)
    n = len(s_clean)
    limit = min(max_trim, n)

    for trim in range(1, limit + 1):
        prefix = s_clean[: n - trim].rstrip()
        if not prefix:
            break
        attempt = _simple_repair(prefix)
        try:
            return json.loads(attempt)
        except Exception:
            continue

    return None  # give up


def parse_or_empty(s: str):
    obj = safe_json_loads_mending(s)
    return obj if obj is not None else {}


id_to_gt = {}
id_to_pred = {}
for item in chartqa_test_data:
    # if "chartqa2table" in item["id"]:
    #     id = item["image"].split("/")[-1]
    #     value = item["conversations"][1]["value"]
    #     if len(value.split("\n")[0].split("|")) == 2:
    #         id_to_gt[id] = "\n".join(value.split("\n")[1:])
    #     else:
    #         id_to_gt[id] = value
    values = item["gts"]["values"]
    if not isinstance(list(values.values())[0], dict):
        rows = [f"{k} | {v}" for k, v in values.items()]
        table = "\n".join(rows)
    else:
        # Transpose the table: rows become columns and vice versa
        col1 = [k for k in values.keys()]
        row_keys = [k for k in values[col1[0]].keys()]
        # Build the transposed table: first column is the header, then each col1 is a row
        header = ["A"] + row_keys
        rows = [header]
        for c in col1:
            row = [c]
            for rk in row_keys:
                row.append(str(values[c].get(rk, "")))
            rows.append(row)
        table = "\n".join([" | ".join(row) for row in rows])
    id_to_gt[item["images"].split(".")[0]] = table

for item in model_predictions:
    # print(item["answer"])
    print(item["imagename"])
    try:
        ans = json.loads(item["answer"])
    except Exception:
        repaired = parse_or_empty(item["answer"])
        ans = repaired  # dict/list or {}
    if not ans or not ans["values"]:
        print("CANNOT PARSE")
        id_to_pred[item["imagename"].split(".")[0]] = ""
        continue
    values = ans["values"]
    if not isinstance(list(values.values())[0], dict):
        rows = [f"{k} | {v}" for k, v in values.items()]
        table = "\n".join(rows)
    else:
        col1 = [k for k in values.keys()]
        row_keys = [k for k in values[col1[0]].keys()]
        header = ["A"] + col1
        rows = [header]
        for rk in row_keys:
            row = [rk]
            for c in col1:
                row.append(str(values[c].get(rk, "")))
            rows.append(row)
        table = "\n".join([" | ".join(row) for row in rows])
    id_to_pred[item["imagename"].split(".")[0]] = table

print("Ground truth length: ", len(id_to_gt))
print("Predictions length: ", len(id_to_pred))
formatted_data = []
for id in id_to_gt:
    if id in id_to_pred:
        formatted_data.append(
            {
                "id": id,
                "question": "<image>\nGenerate underlying data table of the chart.",
                "gt_answer": id_to_gt[id],
                "model_answer": id_to_pred.get(id, ""),
            }
        )

# with open("data/tinychart_formatted_preds_and_gt.json", "w") as f:
#     json.dump(formatted_data, f, indent=4)
# with open("data/tinychart_formatted_preds_and_onechart_gt.json", "w") as f:
with open("data/hard_epicurves_formatted_preds_and_gt.json", "w") as f:
    json.dump(formatted_data, f, indent=4)
