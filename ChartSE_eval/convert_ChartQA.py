# Script to convert ChartQA test data to the format of OneChart

import json

chartqa_test_data = json.load(open("../mPLUG-DocOwl/TinyChart/data/test.json"))

converted_data = []

for item in chartqa_test_data:
    if "chartqa2table" in item["id"]:
        converted_datapoint = {}
        converted_datapoint["images"] = item["image"].split("/")[-1]
        converted_datapoint["gts"] = {
            "title": "None",
            "source": "None",
            "x_title": "None",
            "y_title": "None",
        }

        value = item["conversations"][1]["value"]
        rows = value.split("\n")
        table = [row.split(" | ") for row in rows]
        if len(table[0]) == 2:
            d = {row[0]: row[1] for row in table[1:]}
        else:
            d = {}
            # for r in range(1, len(table)):
            #     row_key = table[r][0]
            #     if row_key not in d:
            #         d[row_key] = {}
            #     for c in range(1, len(table[0])):
            #         d[row_key][table[0][c]] = table[r][c]
            # Transpose the table (excluding header row and column)
            headers = table[0][1:]
            for c in range(1, len(table[0])):
                col_key = table[0][c]
                d[col_key] = {}
                for r in range(1, len(table)):
                    row_key = table[r][0]
                    d[col_key][row_key] = table[r][c]

        converted_datapoint["gts"]["values"] = d

        converted_data.append(converted_datapoint)

with open("data/chartqa_test_converted_transposed.json", "w") as f:
    json.dump(converted_data, f, indent=4)
