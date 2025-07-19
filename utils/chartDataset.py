import torch
from torch.utils.data import Dataset
from consts import note_type_mapping, touch_area_mapping, max_tags


class ChartDataset(Dataset):
    def __init__(self, json_data_list):
        self.data = []
        self.labels = []
        self.metadata = []
        for item in json_data_list:
            notes_sequence = []
            for entry in item["data"]:
                time = entry["Time"]
                for note in entry["Notes"]:
                    note_features = [
                        note["holdTime"],
                        int(note["isBreak"]),
                        int(note["isEx"]),
                        int(note["isFakeRotate"]),
                        int(note["isForceStar"]),
                        int(note["isHanabi"]),
                        int(note["isSlideBreak"]),
                        int(note["isSlideNoHead"]),
                        note_type_mapping[note["noteType"]],
                        note["slideStartTime"],
                        note["slideTime"],
                        note["startPosition"],
                        touch_area_mapping[note["touchArea"]],
                        time,
                        note["density"],
                        note["sweepAllowed"],
                        note["multiPressCount"],
                        note["displacement"],
                    ]
                    notes_sequence.append(note_features)
            self.data.append(notes_sequence)
            probs = [0] * (max_tags + 1)
            for keyword in item["keywords"]:
                probs[int(keyword)] = 1
            self.labels.append(probs)
            # 存储元数据
            self.metadata.append(
                {
                    "song_id": item["song_id"],
                    "level_index": item["level_index"],
                    "keywords_num": len(item["keywords"]),
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.metadata[idx],  # 返回元数据
        )
