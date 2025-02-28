import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

MLP_ENCODER_HIDDEN_DIM = 128
MULTI_TASK_DECODER_HIDDEN_DIM = 32
LATENT_DIM = 32
LATENT_QUERY_DIM = 2


class EncoderType(Enum):
    MLP = 0
    Transformer = 1


class MLPEncoder(nn.Module):

    def __init__(self, feature_len, seq_len):
        super(MLPEncoder, self).__init__()
        # Flatten the input (batch_size, seq_len, feature_len) -> (batch_size, seq_len * feature_len)
        input_dim = feature_len * seq_len

        # Define MLP layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, MLP_ENCODER_HIDDEN_DIM),
            nn.LayerNorm(MLP_ENCODER_HIDDEN_DIM),  # Layer Normalization
            nn.ReLU(),
            nn.Dropout(0.01),  # Dropout with 10% probability
            nn.Linear(MLP_ENCODER_HIDDEN_DIM, LATENT_DIM))

    def forward(self, x):
        x = x.float()
        batch_size = x.shape[0]
        # Flatten input
        x = x.reshape(batch_size, -1)

        # Pass through MLP
        latent_output = self.encoder(x)

        return latent_output


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


# Encoder Model
class LatentQueryTransformerEncoder(nn.Module):

    def __init__(self, feature_len, seq_len, nhead=4, num_layers=2):
        super(LatentQueryTransformerEncoder, self).__init__()

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(feature_len, seq_len)

        # Input projection
        self.input_proj = nn.Linear(feature_len, LATENT_DIM)

        # Latent query (learnable)
        LATENT_QUERY_DIM = 2
        self.latent_queries = nn.Parameter(
            torch.randn(1, LATENT_QUERY_DIM, LATENT_DIM))

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=LATENT_DIM,
                                                    nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

        # Output layer
        self.output_proj = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, x):
        x = x.float()
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]

        # Add positional encoding
        x = self.pos_encoder(x)

        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, latent_dim)

        # Prepare latent query
        latent_queries = self.latent_queries.expand(
            batch_size, -1, -1)  # (batch_size, num_queries, latent_dim)

        # Concatenate latent query at the beginning
        x = torch.cat([latent_queries, x],
                      dim=1)  # (batch_size, num_queries+seq_len, latent_dim)

        # Transformer expects shape (num_queries+seq_len, batch_size, latent_dim)
        x = x.transpose(0, 1)

        # Pass through transformer encoder
        memory = self.transformer_encoder(
            x)  # (num_queries+seq_len, batch_size, latent_dim)

        # Extract only the latent query outputs
        query_output = memory[:
                              LATENT_QUERY_DIM]  # (num_queries, batch_size, latent_dim)
        # Reshape back to (batch_size, num_queries, latent_dim)
        query_output = query_output.transpose(0, 1)

        # Output projection - Aggregate over sequence dimension
        output = self.output_proj(
            query_output.mean(dim=1))  # (batch_size, latent_dim)

        return output


class MultiTaskClassifier(nn.Module):

    def __init__(self):
        super(MultiTaskClassifier, self).__init__()
        self.fc1 = nn.Linear(LATENT_DIM, MULTI_TASK_DECODER_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(MULTI_TASK_DECODER_HIDDEN_DIM
                                )  # Normalizes across feature dimensions
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.01)  # 1% Dropout
        self.out = nn.Linear(MULTI_TASK_DECODER_HIDDEN_DIM, 6)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        logits = self.out(x)  # Raw logits for BCEWithLogitsLoss

        # Return raw logits (use CrossEntropyLoss directly)
        return logits


class PredictionModel(nn.Module):

    def __init__(self,
                 feature_len,
                 seq_len,
                 encoder_type=EncoderType.Transformer):
        super(PredictionModel, self).__init__()
        if encoder_type == EncoderType.MLP:
            self.encoder_model = MLPEncoder(feature_len=feature_len,
                                            seq_len=seq_len)
        elif encoder_type == EncoderType.Transformer:
            self.encoder_model = LatentQueryTransformerEncoder(feature_len=feature_len,
                                                    seq_len=seq_len)
        else:
            raise ValueError("Invalid encoder type")

        self.output_model = MultiTaskClassifier()

    def forward(self, x):
        embedding = self.encoder_model(x)
        output = self.output_model(embedding)

        return output


if __name__ == "__main__":
    # Example usage
    encoder_model = LatentQueryTransformerEncoder(feature_len=11, seq_len=30)
    # encoder_model = MLPEncoder(feature_len=11, seq_len=30)

    # Example input: batch_size=16
    example_input = torch.randn(16, 30, 11)
    embedding = encoder_model(example_input)
    print(embedding.shape)  # Should print: torch.Size([16, 32])

    output_model = MultiTaskClassifier()
    output = output_model(embedding)
    # print(output.shape)
