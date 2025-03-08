import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import PositionalEncoding, AttentionPooling
from data import label
from config.config import EncoderType, device

MLP_ENCODER_HIDDEN_DIM = 128
MULTI_TASK_DECODER_HIDDEN_DIM = 64
LATENT_DIM = 64
LATENT_QUERY_DIM = 2


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


class TransformerEncoder(nn.Module):

    def __init__(self, feature_len, seq_len, nhead=4, num_layers=2):
        super(TransformerEncoder, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(feature_len, LATENT_DIM)
        self.ln = nn.LayerNorm(
            LATENT_DIM)  # Normalizes across feature dimensions

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(LATENT_DIM, seq_len)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=LATENT_DIM,
                                                    nhead=nhead,
                                                    dim_feedforward=4 *
                                                    LATENT_DIM,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

        self.attention_pooling = AttentionPooling(LATENT_DIM)

        # Output layer
        self.output_proj = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, x):
        x = x.float()  # x shape: (batch_size, seq_len, input_dim)
        # batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, latent_dim)
        x = self.ln(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        memory = self.transformer_encoder(
            x)  # (batch_size, seq_len,  latent_dim)

        # Output projection - Aggregate over sequence dimension
        output = self.output_proj(
            self.attention_pooling(memory))  # (batch_size, latent_dim)

        return output


# Transformer Encoder Model with latent query
class LatentQueryTransformerEncoder(nn.Module):

    def __init__(self, feature_len, seq_len, nhead=4, num_layers=4):
        super(LatentQueryTransformerEncoder, self).__init__()

        # Input projection
        self.input_proj = nn.Linear(feature_len, LATENT_DIM)
        self.ln = nn.LayerNorm(
            LATENT_DIM)  # Normalizes across feature dimensions

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(LATENT_DIM, seq_len)

        # Latent query (learnable)
        self.latent_queries = nn.Parameter(
            torch.randn(1, LATENT_QUERY_DIM, LATENT_DIM))

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=LATENT_DIM,
                                                    nhead=nhead,
                                                    dim_feedforward=4 *
                                                    LATENT_DIM,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers,
                                                         num_layers=num_layers)

        # Output layer
        self.output_proj = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, x):
        x = x.float()
        # x shape: (batch_size, seq_len, input_dim)
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, latent_dim)
        x = self.ln(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Prepare latent query
        latent_queries = self.latent_queries.expand(
            batch_size, -1, -1)  # (batch_size, num_queries, latent_dim)

        # Concatenate latent query at the beginning
        x = torch.cat([latent_queries, x],
                      dim=1)  # (batch_size, num_queries+seq_len, latent_dim)

        # Pass through transformer encoder
        memory = self.transformer_encoder(
            x)  # (batch_size, num_queries+seq_len, latent_dim)

        # Extract only the latent query outputs
        query_output = memory[:, :
                              LATENT_QUERY_DIM, :]  # (batch_size, num_queries, latent_dim)

        # Output projection - Aggregate over sequence dimension
        output = self.output_proj(
            query_output.mean(dim=1))  # (batch_size, latent_dim)

        return output


class DualAttentionTransformerEncoder(nn.Module):

    def __init__(self, feature_len, seq_len, num_heads=4, num_layers=2):
        super(DualAttentionTransformerEncoder, self).__init__()

        # Project input to latent space
        self.input_proj = nn.Linear(feature_len, LATENT_DIM)
        self.ln = nn.LayerNorm(
            LATENT_DIM)  # Normalizes across feature dimensions

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(LATENT_DIM, seq_len)

        # Self-Attention on time steps
        self.time_attention = nn.MultiheadAttention(embed_dim=LATENT_DIM,
                                                    num_heads=num_heads,
                                                    batch_first=True)

        # Self-Attention on feature dimensions
        self.feature_attention = nn.MultiheadAttention(embed_dim=seq_len,
                                                       num_heads=6,
                                                       batch_first=True)

        # Cross-Attention on time-feature dimensions
        self.cross_attention = nn.MultiheadAttention(embed_dim=LATENT_DIM,
                                                     num_heads=num_heads,
                                                     batch_first=True)

        # Transformer Encoder Layer (for multi-head time and feature attention)
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=LATENT_DIM,
                                                         nhead=num_heads,
                                                         dim_feedforward=4 *
                                                         LATENT_DIM,
                                                         batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers,
                                                         num_layers=num_layers)

        self.attention_pooling = AttentionPooling(LATENT_DIM)

        # Output layer
        self.output_proj = nn.Linear(LATENT_DIM, LATENT_DIM)

    def forward(self, x):
        x = x.float()
        # x shape: (batch_size, seq_len, input_dim)

        # Step 1: Project input to latent space
        x = self.input_proj(x)  # Shape: (batch_size, seq_len, latent_dim)
        x = self.ln(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Step 2: Apply self-attention on time steps
        x_time, _ = self.time_attention(
            x, x, x)  # Query, Key, Value are all x (time-step attention)
        # (batch_size, seq_len, latent_dim)

        # Step 3: Apply self-attention on feature dimensions (transpose for feature attention)
        x_feature_input = x.transpose(
            1, 2)  # Shape: (batch_size, latent_dim, seq_len)
        x_feature, _ = self.feature_attention(x_feature_input, x_feature_input,
                                              x_feature_input)
        # Transpose back to (batch_size, seq_len, latent_dim)
        x_feature = x_feature.transpose(1, 2)

        cross_output, _ = self.cross_attention(x_time, x_feature, x_feature)

        # Step 4: Pass through Transformer Encoder (can model both time-step and feature attention)
        memory = self.transformer_encoder(cross_output)

        # Step 5: Final projection to output feature dimension - # Aggregate over sequence dimension
        output = self.output_proj(
            self.attention_pooling(memory))  # (batch_size, latent_dim)

        return output


class MultiTaskClassifier(nn.Module):

    def __init__(self):
        super(MultiTaskClassifier, self).__init__()
        self.fc1 = nn.Linear(LATENT_DIM, MULTI_TASK_DECODER_HIDDEN_DIM)
        self.ln1 = nn.LayerNorm(MULTI_TASK_DECODER_HIDDEN_DIM
                                )  # Normalizes across feature dimensions
        self.fc2 = nn.Linear(MULTI_TASK_DECODER_HIDDEN_DIM,
                             MULTI_TASK_DECODER_HIDDEN_DIM)
        self.ln2 = nn.LayerNorm(MULTI_TASK_DECODER_HIDDEN_DIM
                                )  # Normalizes across feature dimensions
        self.dropout = nn.Dropout(p=0.1)  # 1% Dropout
        self.out = nn.Linear(MULTI_TASK_DECODER_HIDDEN_DIM, 6)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
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
            self.encoder_model = TransformerEncoder(feature_len=feature_len,
                                                    seq_len=seq_len)
        elif encoder_type == EncoderType.LatentQueryTransformer:
            self.encoder_model = LatentQueryTransformerEncoder(
                feature_len=feature_len, seq_len=seq_len)
        elif encoder_type == EncoderType.DualAttentionTransformer:
            self.encoder_model = DualAttentionTransformerEncoder(
                feature_len=feature_len, seq_len=seq_len)
        else:
            raise ValueError("Invalid encoder type")

        self.output_model = MultiTaskClassifier()

    def forward(self, x):
        embedding = self.encoder_model(x)
        output = self.output_model(embedding)

        return output


class CustomLoss(nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()
        self.class_weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0,
                                           3.0]).to(device)
        # Add higher weight to positive class
        positive_weights = torch.tensor([3.0]).to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(self.class_weights,
                                             pos_weight=positive_weights)

    def forward(self, logits, targets):
        loss = self.bce_loss(logits, targets)

        # Negative labels for (+, -) are the same, so we can use contrastive loss
        alpha = 1.0
        for i in range(0, len(label.label_feature), 2):
            # Extract logits for class 0 and class 1
            logit_up = logits[:, i]
            logit_down = logits[:, i + 1]

            # Mask: Only consider cases where both labels are negative (0)
            mask = (targets[:, i] == 0) & (targets[:, i + 1] == 0)

            # Contrastive loss: minimize L2 distance when both labels are negative
            contrastive_loss = torch.mean(
                (logit_up[mask] - logit_down[mask])**2) if mask.any() else 0.0

            # Combine losses
            loss += alpha * self.class_weights[i] * contrastive_loss

        # Positive labels for (+, -) should not be the same, so add contrastive loss
        margin = 0.5
        for i in range(0, len(label.label_feature), 2):
            # Extract logits for class 0 and class 1
            logit_up = logits[:, i]
            logit_down = logits[:, i + 1]

            # Mask: Only consider cases where either label is positive (1)
            mask = (targets[:, i] == 1) | (targets[:, i + 1] == 1)

            # Compute distance between logit_up and logit_down
            distance = torch.abs(
                logit_up - logit_down
            )  #F.pairwise_distance(logit_up.unsqueeze(1), logit_down.unsqueeze(1), p=2)

            # Compute loss only for masked samples
            contrastive_loss = torch.mean(
                mask.float() *
                F.relu(margin - distance))  # Ensuring minimum separation

            # Combine losses
            loss += alpha * self.class_weights[i] * contrastive_loss

        return loss


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
