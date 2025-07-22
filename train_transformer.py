import csv
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from consts import difficulty_mapping, max_tags, special_indices
from tqdm import tqdm
import os
from models.Transformer2 import TransformerWithHead
from utils.chartDataset import ChartDataset
import wandb
import swanlab

swanlab.sync_wandb()

# 添加损失热力图可视化
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["WANDB_MODE"] = "offline"

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

note_dataset = ChartDataset(json_data_list)
train_size = int(len(note_dataset) * 0.9)
test_size = len(note_dataset) - train_size
train_dataset, test_dataset = random_split(note_dataset, [train_size, test_size])

max_note_length = max(len(sequence) for sequence in note_dataset.data)
print(f"数据集中最大 note 数量: {max_note_length}")

# 使用训练集中最大 note 数量 + 200 作为固定长度，一般地，这个值通常在 1600 左右
fixed_note_length = 1600


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


# 创建训练和测试数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)


model = TransformerWithHead(
    input_dim=18,
    d_model=128,
    num_heads=16,
    num_layers=2,
    num_labels=max_tags + 1,  # 根据你的数据
    max_len=fixed_note_length,
    dropout=0.1,
).cuda()

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, "min", patience=3, factor=0.5
)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# # 固定步长衰减
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

# 定义早停机制
early_stopping_patience = 10
best_loss = float("inf")
epochs_no_improve = 0

# 定义目标损失值
target_loss = 0.01

run = wandb.init(
    entity="dusker233-southeastern-university",
    project="maimai-chart-keyword-predictor",
    config={
        "learning_rate": 1e-4,
        # "momentum": 0.9,
        "batch_size": 32,
        "epochs": 100,
        # "early_stopping_patience": 10,
        # "target_loss": 0.01,
        "model_name": "Transformer",
        "optimizer": "AdamW",
        "model.hidden_size": 128,
        "model.num_layers": 2,
        "model.num_heads": 16,
        "model.dropout": 0.1,
        # "model.special_weight": 5.0,
        # "optimizer": "SGD",
        # "scheduler": "StepLR",
        # "scheduler.step_size": 15,
        # "scheduler.gamma": 0.1,
    },
)
run.watch(model, log_freq=1)
run_id = run.id

# 检查是否存在已保存的模型
if not os.path.exists(f"./trained_models/{run_id}"):
    os.makedirs(f"./trained_models/{run_id}", exist_ok=True)
model_path = f"trained_models/{run_id}/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"加载已保存的模型: {model_path}")
else:
    print("未找到已保存的模型，从头开始训练")

# 训练模型
num_epochs = 100  # 设置一个较大的初始值
save_interval = 10  # 每10轮保存一次模型


def calculate_accuracy(outputs, labels, threshold=0.5):
    """计算多标签分类准确率"""
    probs = torch.sigmoid(outputs)  # 转换为概率
    predicted = (probs >= threshold).float()  # 二值化预测结果
    # print(predicted, labels)
    correct = (predicted == labels).sum().item()
    total = labels.numel()
    return correct / total


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            # 计算准确率
            correct += ((torch.sigmoid(outputs) >= 0.5).float() == labels).sum().item()
            total += labels.numel()
    return correct / total


from sklearn.metrics import f1_score


def calculate_f1(outputs, labels, threshold=0.5):
    probs = torch.sigmoid(outputs).cpu().detach().numpy()
    preds = (probs >= threshold).astype(int)
    return f1_score(labels.cpu().detach().numpy().flatten(), preds.flatten(), average='weighted')


def calculate_eval_f1(model, test_loader):
    model.eval()
    f1 = 0
    for inputs, labels, _ in test_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        f1 += calculate_f1(outputs, labels)
    return f1 / len(test_loader)


for epoch in tqdm(range(num_epochs)):
    model.train()
    epoch_loss = 0
    train_acc = 0
    train_f1 = 0
    high_error_samples = []  # 存储高误差样本

    for inputs, labels, metadata in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        raw_losses = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs,
            labels,
            reduction="none",
        )
        batch_losses = raw_losses.sum(dim=1)

        threshold = batch_losses.mean()
        high_error_mask = batch_losses >= threshold
        high_error_indices = high_error_mask.nonzero().flatten().tolist()

        for idx in high_error_indices:
            probbs = torch.sigmoid(outputs[idx])
            pred = (probbs >= 0.5).float()
            high_error_samples.append(
                [
                    metadata[idx]["song_id"],
                    metadata[idx]["level_index"],
                    batch_losses[idx].item(),
                    pred.tolist(),
                    labels[idx].tolist(),
                ]
            )
        # 使用 wandb 可视化每个标签的损失分布
        # for tag_idx in range(25):
        #     run.log(
        #         {
        #             f"loss_tag_{tag_idx}": wandb.Histogram(
        #                 raw_losses[:, tag_idx].detach().cpu().numpy()
        #             )
        #         },
        #         step=epoch,
        #     )
        # plt.figure(figsize=(10, 6))
        # sns.heatmap(raw_losses.cpu().detach().numpy(), cmap="viridis")
        # run.log({"loss_heatmap": wandb.Image(plt)}, step=epoch)
        # plt.close()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        train_acc += calculate_accuracy(outputs, labels)
        train_f1 += calculate_f1(outputs, labels)
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    epoch_loss /= len(train_loader)
    train_acc /= len(train_loader)
    train_f1 /= len(train_loader)
    test_acc = evaluate(model, test_loader)
    test_f1 = calculate_eval_f1(model, test_loader)

    # scheduler.step()
    scheduler.step(epoch_loss)

    run.log(
        {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        },
        step=epoch,
    )
    high_error_samples = sorted(high_error_samples, key=lambda x: x[2], reverse=True)
    high_error_samples = high_error_samples[:5]
    columns = ["song_id", "level_index", "loss", "prediction", "label"]
    wandb_table = wandb.Table(columns=columns, data=high_error_samples)
    swanlab_table = swanlab.echarts.Table()
    swanlab_table.add(headers=columns, rows=high_error_samples)
    run.log({"high_error_samples": wandb_table}, step=epoch)
    swanlab.log({"high_error_samples": swanlab_table})

    # 打印高误差样本
    print(
        f"\nEpoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}"
    )

    # 保存模型
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(
            "trained_models", f"{run_id}", f"model_epoch_{epoch+1}.pth"
        )
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存: {model_save_path}")

    # 检查当前损失值并与 best 比对
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        epochs_no_improve = 0
        # 保存最好的模型
        best_model_path = f"trained_models/{run_id}/best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        print(f"最好的模型已保存: {best_model_path}")
    else:
        epochs_no_improve += 1

    # if epochs_no_improve >= early_stopping_patience:
    #     print("早停机制触发，停止训练")
    #     break

    # 检查是否达到目标损失值
    if epoch_loss <= target_loss:
        print(f"达到目标损失值 {target_loss}，停止训练")
        break

run.finish()
print("训练完成")
