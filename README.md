[![](https://raw.githubusercontent.com/SwanHubX/assets/main/badge1.svg)](https://swanlab.cn/@Dusker233/maimai-chart-keyword-predictor/overview)

XgBoost 基准分类：acc=0.0360

BiLSTM: acc=0.812
Transformer: acc=0.8151
LSTM: acc=0.8022

## highest error sample when training

|Song_id|Level_index|Loss|Prediction|Label|Method|
|---|---|---|---|---|---|
|11663|4|26.32314682006836|[ 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ]|[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]|BiLSTM|
|11663|4|31.803020477294922|[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]|[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]|Transformer|
|11663|4|26.626995086669922|[ 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ]|[ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 ]|LSTM|