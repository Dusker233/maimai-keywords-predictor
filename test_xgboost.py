import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
from torch.utils.data import DataLoader
from utils.chartDataset import ChartDataset
import torch
from collections import defaultdict
import csv
import json
from tqdm import tqdm
from consts import difficulty_mapping, max_tags, special_indices
import os

# 1️⃣ 加载 ChartDataset
csv_file_path = "./info/chart_info.csv"
title_id_path = "./title_id.csv"
tags_path = "./combined-tags-mapped.json"
dataset_path = "./"

id_title_map = defaultdict(str)
id_keywords_map = defaultdict(list)
json_data_list = []

with open(title_id_path, "r", encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in tqdm(csv_reader):
        id = row["ID"]
        title = row["name"]
        type = row["type"]
        id_title_map[(title, type)] = id

with open(tags_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)
    taged_songs = json_data["tagSongs"]
    for tag_song in tqdm(taged_songs):
        title = tag_song["song_id"]
        type = tag_song["sheet_type"]
        id = id_title_map[(title, type)]
        difficulty = (
            difficulty_mapping[tag_song["sheet_difficulty"]]
            if tag_song["sheet_difficulty"] in difficulty_mapping
            else 5
        )
        id_keywords_map[id].append((difficulty, tag_song["tag_id"]))

mx = 0
for id, tags in id_keywords_map.items():
    for tag in tags:
        mx = max(mx, tag[1])
print(f"最大标签id: {mx}")

with open(csv_file_path, "r", encoding="utf-8") as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        # print(row)
        # 获取 JSON 文件路径
        json_file_path = os.path.join(dataset_path, row["FilePath"])

        # 读取 JSON 文件
        # print(json_file_path)
        with open(json_file_path, mode="r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
            song_id, diff = row["ID"], row["Difficulty"]
            tags = []
            for d, tag in id_keywords_map[song_id]:
                if d == int(diff):
                    tags.append(tag)
            if tags == []:
                continue
            # 将 JSON 数据添加到列表中
            json_data_list.append(
                {
                    "song_id": row["ID"],
                    "level_index": row["Difficulty"],
                    "data": json_data,
                    "keywords": tags,
                }
            )

print(f"读取了 {len(json_data_list)} 个 JSON 文件的数据")

train_data = ChartDataset(json_data_list)

# 2️⃣ 从 Dataset 提取 X, y
X_list = []
y_list = []

for i in range(len(train_data)):
    x, y, _ = train_data[i]   # x: (seq_len, 18), y: (num_labels,)
    x = x.mean(dim=0)         # 🧠 平均池化 → x: (18,)
    X_list.append(x.numpy())
    y_list.append(y.numpy())

X = np.stack(X_list)          # shape: (num_samples, 18)
y = np.stack(y_list)          # shape: (num_samples, num_labels)

# 3️⃣ 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 多标签分类，每个标签一个 XGBClassifier
models = []
y_preds = []

for i in range(y.shape[1]):
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=True,
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train[:, i])
    pred = clf.predict(X_test)
    models.append(clf)
    y_preds.append(pred)

# 5️⃣ 汇总评估
y_preds = np.stack(y_preds, axis=1)
f1 = f1_score(y_test, y_preds, average='micro')
acc = accuracy_score(y_test, y_preds)

print(f"🎯 XGBoost F1（micro）: {f1:.4f}")
print(f"✅ XGBoost 准确率: {acc:.4f}")
