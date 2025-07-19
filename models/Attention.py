import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size, special_indices, special_weight=10.0):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.context_vector = nn.Linear(hidden_size, 1, bias=False)
        self.special_indices = special_indices
        self.special_weight = special_weight

    def forward(self, lstm_output):
        # lstm_output: [batch_size, seq_len, hidden_size]
        attention_weights = torch.tanh(self.attention(lstm_output))
        attention_weights = self.context_vector(attention_weights).squeeze(-1)

        # 增加对特定参数的关注
        special_attention = lstm_output[:, :, self.special_indices].sum(dim=-1)
        attention_weights += self.special_weight * special_attention

        attention_weights = torch.softmax(attention_weights, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context_vector, attention_weights
