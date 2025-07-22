import torch
import torch.nn as nn

class TransformerWithHead(nn.Module):
    def __init__(self, input_dim=18, d_model=256, num_heads=8, num_layers=4, num_labels=512, max_len=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, input_dim)
        B, L, _ = x.size()
        x = self.input_proj(x) + self.pos_embedding[:, :L, :]
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Mean Pooling
        logits = self.classifier(x)
        return logits
        # return torch.sigmoid(logits)
