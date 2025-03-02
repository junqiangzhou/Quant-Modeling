import torch
import math
import torch.nn as nn


# Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, feature_len: int, seq_len: int):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(seq_len, feature_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, feature_len, 2).float() *
            (-math.log(10000.0) / feature_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        if feature_len % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class AttentionPooling(nn.Module):

    def __init__(self, latent_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(
            latent_dim, 1)  # Learnable weights for each time step

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, latent_dim) - Transformer outputs
        Returns:
            pooled_output: (batch_size, latent_dim) - Weighted representation
        """
        attn_scores = self.attention_weights(x).squeeze(
            -1)  # (batch_size, seq_len)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(
            -1)  # (batch_size, seq_len, 1)

        pooled_output = (x * attn_weights).sum(
            dim=1)  # Weighted sum over time steps
        return pooled_output
