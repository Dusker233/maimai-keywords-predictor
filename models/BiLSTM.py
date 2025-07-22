import torch
import torch.nn as nn
from .Attention import Attention  # 复用项目中的 Attention 模块

class BiLSTMWithAttention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        output_size,
        special_indices,
        special_weight=10.0,
    ):
        super(BiLSTMWithAttention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attention = Attention(
            hidden_size=hidden_size * 2,  # 双向
            special_indices=special_indices,
            special_weight=special_weight,
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c_0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.bilstm(x, (h_0, c_0))  # (B, L, 2*H)
        context_vector, attention_weights = self.attention(lstm_out)
        output = self.fc(context_vector).squeeze(-1)
        return output, attention_weights
