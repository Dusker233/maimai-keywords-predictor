# 定义 NoteType 和 TouchArea 的映射
note_type_mapping = {"Tap": 0, "Slide": 1, "Hold": 2, "Touch": 3, "TouchHold": 4}
touch_area_mapping = {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}

# 特定参数的索引
special_indices = [
    13,
    14,
    15,
    16,
    17,
]  # 对应 note["time"], note["density"], note["sweepAllowed"], note["multiPressCount"], note["displacement"]

difficulty_mapping = {
    "basic": 0,
    "advanced": 1,
    "expert": 2,
    "master": 3,
    "remaster": 4,
}

max_tags = 20
