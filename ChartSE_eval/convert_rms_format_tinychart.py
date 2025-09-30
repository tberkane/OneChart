# Script to convert the format of the ground truth and predictions to the same format as TinyChart
# such that the TinyChart code for evaluation (using RMS metric) can be used.

import json

chartqa_test_data = json.load(open("data/EpiCurveBench_processed/gt.json"))
model_predictions = json.load(
    open("data/EpiCurveBench_processed/tinychart_adv_preds.json")
)

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
id_to_pred_reversed = {}
id_to_axes_bounds = {}
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
    id_to_axes_bounds[item["images"].split(".")[0]] = item["axes_bounds"]

for item in model_predictions:
    # print(item["answer"])
    print(item["imagename"])
    if item["answer"].count("\n") == 1:
        row1, row2 = item["answer"].split("\n")
        item["answer"] = "\n".join(
            [col1 + "|" + col2 for col1, col2 in zip(row1.split("|"), row2.split("|"))]
        )

    if (
        item["answer"].split("\n")[0].count("|") == 1
        and id_to_gt[item["imagename"].split(".")[0]].split("\n")[0].count("|") == 1
    ):
        pred = "\n".join(item["answer"].split("\n")[1:])
        pred_reversed = "\n".join(pred.split("\n")[::-1])
    elif (
        item["answer"].split("\n")[0].count("|") > 1
        and id_to_gt[item["imagename"].split(".")[0]].split("\n")[0].count("|") > 1
    ):
        changed = "A |" + item["answer"].split("|", 1)[1]
        pred = changed
        pred_split = pred.split("\n")
        pred_reversed = "\n".join([pred_split[0]] + pred_split[1:][::-1])
    else:
        pred = item["answer"]
        pred_split = pred.split("\n")
        pred_reversed = "\n".join([pred_split[0]] + pred_split[1:][::-1])
    if item["answer"].split("\n")[0].count("|") == 1:
        if pred.split("\n")[-1].count("|") != 1:
            pred = "\n".join(pred.split("\n")[:-1])
            pred_reversed = "\n".join(pred_reversed.split("\n")[1:])
    id_to_pred[item["imagename"].split(".")[0]] = pred
    id_to_pred_reversed[item["imagename"].split(".")[0]] = pred_reversed

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
                "model_answer_reversed": id_to_pred_reversed.get(id, ""),
                "axes_bounds": id_to_axes_bounds[id],
            }
        )

formatted_data.sort(key=lambda x: int(x["id"]))


with open("data/EpiCurveBench_processed/tinychart_adv_preds_gt.json", "w") as f:
    json.dump(formatted_data, f, indent=4)
