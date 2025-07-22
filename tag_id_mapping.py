import json
from collections import defaultdict

with open("./combined-tags.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    tags = data["tags"]
    cnt = 0
    id_cnt_mapping = defaultdict(int)
    for tag in tags:
        id = tag["id"]
        if id_cnt_mapping[id] == 0:
            cnt += 1
            id_cnt_mapping[id] = cnt
    for key in id_cnt_mapping:
        id_cnt_mapping[key] -= 1
    for tag in data["tags"]:
        tag["id"] = id_cnt_mapping[tag["id"]]
    for tag in data["tagSongs"]:
        tag["tag_id"] = id_cnt_mapping[tag["tag_id"]]

with open("./combined-tags-mapped.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
