import json
import os

for datasets in ["train", "test", "val"]:

    s = ""

    for file_name in os.listdir(r"datasets/jsons"):
        json_path = os.path.join(r"datasets/jsons", file_name)

        if datasets in file_name:

            data = json.load(open(json_path, 'r', encoding='utf-8'), strict=False)
            s = s + "{} {} {} {} {} {} {}\n".format(file_name.replace(".json", ".jpg"), data["start"][0], data["start"][1],
                                                   data["medium"][0], data["medium"][1], data["end"][0], data["end"][1])

    f = open(r"datasets/annotations/{}_dataset.txt".format(datasets), "w", encoding="utf-8")
    f.write(s)
    f.close()
