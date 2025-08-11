# Script to convert the format of the ground truth and predictions to the same format as TinyChart
# such that the TinyChart code for evaluation (using RMS metric) can be used.

import json

# chartqa_test_data = json.load(open("../mPLUG-DocOwl/TinyChart/data/test.json"))
chartqa_test_data = json.load(open("data/ChartQA_test_human_filter.json"))
model_predictions = json.load(open("data/example_pred(use_this).json"))

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
    id_to_gt[item["images"]] = table

for item in model_predictions:
    # print(item["answer"])
    try:
        ans = json.loads(item["answer"])
    except Exception as e:
        print(e)
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
    id_to_pred[item["imagename"]] = table

print("Ground truth length: ", len(id_to_gt))
print("Predictions length: ", len(id_to_pred))

formatted_data = []
for id in id_to_gt:
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
with open("data/tinychart_formatted_preds_and_onechart_gt.json", "w") as f:
    json.dump(formatted_data, f, indent=4)
