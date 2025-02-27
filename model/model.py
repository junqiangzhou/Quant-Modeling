import torch
import torch.nn as nn
import torch.nn.functional as F


# Positional Encoding class
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# Encoder Model
class EncoderModel(nn.Module):

    def __init__(self,
                 input_dim=9,
                 seq_len=30,
                 latent_dim=32,
                 nhead=4,
                 num_layers=2):
        super(EncoderModel, self).__init__()

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(latent_dim, max_len=seq_len)

        # Latent query (learnable)
        self.latent_query = nn.Parameter(torch.randn(1, 1, latent_dim))

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                    nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

        # Output layer
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, src):
        src = src.float()
        # src shape: (batch_size, seq_len, input_dim)
        batch_size = src.shape[0]

        # Project input
        src = self.input_proj(src)  # (batch_size, seq_len, latent_dim)

        # Add positional encoding
        src = self.pos_encoder(src)

        # Prepare latent query
        latent_query = self.latent_query.repeat(
            batch_size, 1, 1)  # (batch_size, 1, latent_dim)

        # Concatenate latent query at the beginning
        src = torch.cat([latent_query, src],
                        dim=1)  # (batch_size, seq_len+1, latent_dim)

        # Transformer expects shape (seq_len+1, batch_size, latent_dim)
        src = src.transpose(0, 1)

        # Pass through transformer encoder
        memory = self.transformer_encoder(
            src)  # (seq_len+1, batch_size, latent_dim)

        # Extract latent query output (first token)
        latent_output = memory[0]  # (batch_size, latent_dim)

        # Output projection
        output = self.output_proj(latent_output)  # (batch_size, latent_dim)

        return output


class MultiTaskModel(nn.Module):

    def __init__(self, input_dim, hidden_dim=64):
        super(MultiTaskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(
            hidden_dim)  # Normalizes across feature dimensions
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.01)  # 1% Dropout
        self.out = nn.Linear(hidden_dim, 6)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        logits = self.out(x)  # Raw logits for BCEWithLogitsLoss

        # Return raw logits (use CrossEntropyLoss directly)
        return logits


class PredictionModel(nn.Module):

    def __init__(self, input_dim, seq_len, latent_dim=32, hidden_dim=64):
        super(PredictionModel, self).__init__()
        self.encoder_model = EncoderModel(input_dim=input_dim,
                                          seq_len=seq_len,
                                          latent_dim=latent_dim)
        self.output_model = MultiTaskModel(input_dim=latent_dim,
                                           hidden_dim=hidden_dim)

    def forward(self, x):
        embedding = self.encoder_model(x)
        output = self.output_model(embedding)

        return output


if __name__ == "__main__":
    # Example usage
    encoder_model = EncoderModel(input_dim=11, seq_len=30, latent_dim=32)

    # Example input: batch_size=16
    example_input = torch.randn(16, 30, 11)
    embedding = encoder_model(example_input)
    print(embedding.shape)  # Should print: torch.Size([16, 32])

    output_model = MultiTaskModel(input_dim=32, hidden_dim=32)
    output = output_model(embedding)
    # print(output.shape)
