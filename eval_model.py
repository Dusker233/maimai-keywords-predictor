"""
eval_model.py - 模型评估脚本
功能：
1. 加载训练好的LSTM模型
2. 评估新数据集
3. 输出准确率/F1分数/高误差样本
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.LSTM import LSTMWithAttention
from consts import difficulty_mapping, max_tags, special_indices
from sklearn.metrics import f1_score

from utils.chartDataset import ChartDataset

# 配置参数
MODEL_PATH = "trained_models/7t82x81z/best_model.pth"  # 模型路径
DATA_PATH = "./new_data"  # 新数据路径
BATCH_SIZE = 32
THRESHOLD = 0.5


def load_model():
    """加载训练好的模型"""
    model = LSTMWithAttention(
        input_size=18,
        hidden_size=128,
        num_layers=2,
        output_size=max_tags + 1,
        special_indices=special_indices,
        special_weight=10.0,
    ).cuda()

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"✅ 成功加载模型: {MODEL_PATH}")
    else:
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 未找到")

    model.eval()
    return model


def process_new_data():
    """处理新数据"""
    # 假设新数据格式与训练数据相同
    json_data_list = []

    # 读取新数据
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_PATH, filename), "r") as f:
                json_data = json.load(f)
                filename = filename.replace(".json", "")
                song_id, level = filename.split("_")
                json_data_list.append(
                    {
                        "song_id": song_id,
                        "level_index": int(level),
                        "data": json_data,
                        "keywords": [],
                    }
                )

    print(f"📁 加载了 {len(json_data_list)} 个新数据样本")
    return json_data_list

fixed_note_length = 2000

def collate_fn(batch):
    data, labels, metadata = zip(*batch)

    # 处理 data
    data_padded = pad_sequence(data, batch_first=True)
    if data_padded.size(1) > fixed_note_length:
        data_padded = data_padded[:, :fixed_note_length, :]
    else:
        padding_size = fixed_note_length - data_padded.size(1)
        data_padded = torch.nn.functional.pad(data_padded, (0, 0, 0, padding_size))

    # 处理 labels
    labels_padded = torch.stack(labels)

    return data_padded, labels_padded, metadata

def evaluate(model, data_loader):
    """评估模型表现"""
    all_probs = []
    all_preds = []
    all_labels = []
    all_metadata = []

    with torch.no_grad():
        for inputs, labels, metadata in data_loader:
            inputs = inputs.cuda()
            outputs, _ = model(inputs)

            probs = torch.sigmoid(outputs).cpu()
            # 找出 probs top5 的索引
            preds = probs.topk(10)[1]
            # preds = (probs >= THRESHOLD).float()

            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            if labels is not None:
                all_labels.extend(labels.numpy())
            all_metadata.extend(metadata)

    return np.array(all_probs), np.array(all_preds), np.array(all_labels), all_metadata


def calculate_metrics(probs, labels):
    """计算评估指标"""
    preds = (probs >= THRESHOLD).astype(int)
    flat_preds = preds.flatten()
    flat_labels = labels.flatten()

    accuracy = np.mean(flat_preds == flat_labels)
    f1 = f1_score(flat_labels, flat_preds, average="micro")

    return {"accuracy": accuracy, "f1_score": f1}


def find_high_error_samples(probs, labels, metadata, top_n=5):
    """查找高误差样本"""
    losses = -np.mean(
        labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8), axis=1
    )

    high_error_indices = np.argsort(losses)[-top_n:][::-1]
    high_error_samples = []

    for idx in high_error_indices:
        high_error_samples.append(
            {
                "metadata": metadata[idx],
                "loss": losses[idx],
                "prediction": probs[idx].tolist(),
                "label": labels[idx].tolist(),
            }
        )

    return high_error_samples


def main():
    """主函数"""
    # 1. 加载模型
    model = load_model()

    # 2. 处理新数据
    data_list = process_new_data()
    note_dataset = ChartDataset(data_list)

    # 3. 创建数据加载器
    test_loader = DataLoader(
        note_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 4. 进行评估
    probs, preds, labels, metadata = evaluate(model, test_loader)
    print(f"预测结果：{preds}")

    # 5. 计算指标
    metrics = calculate_metrics(probs, labels)
    print("\n📊 评估结果:")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1分数: {metrics['f1_score']:.4f}")

    # 6. 分析高误差样本
    high_error_samples = find_high_error_samples(probs, labels, metadata)
    print("\n⚠️ 高误差样本:")
    for sample in high_error_samples:
        print(f"ID: {sample['metadata']['song_id']}")
        print(f"Loss: {sample['loss']:.4f}")
        print(f"预测: {sample['prediction']}")
        print(f"标签: {sample['label']}\n")


if __name__ == "__main__":
    main()
