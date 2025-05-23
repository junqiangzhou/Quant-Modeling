from model.utils import PositionalEncoding, AttentionPooling
from config.config import EncoderType, device, label_names
from strategy.rule_based import calc_pred_labels

import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
from numpy.typing import NDArray

MLP_ENCODER_HIDDEN_DIM = 128
MULTI_TASK_DECODER_HIDDEN_DIM = 256
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
        seq_len = x.shape[1]

        # Project input
        x = self.input_proj(x)  # (batch_size, seq_len, latent_dim)
        x = self.ln(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask (upper triangular mask with boolean dtype)
        causal_mask = torch.triu(torch.ones(seq_len,
                                            seq_len,
                                            dtype=torch.bool,
                                            device=x.device),
                                 diagonal=1)

        # Pass through transformer encoder
        memory = self.transformer_encoder(
            x, mask=causal_mask)  # (batch_size, seq_len,  latent_dim)

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
        self.dropout = nn.Dropout(p=0.1)  # 10% Dropout
        self.out = nn.Linear(MULTI_TASK_DECODER_HIDDEN_DIM,
                             3 * len(label_names))

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
        # Add different weights to each class
        self.class_weights = torch.tensor([0.5, 1.0, 1.0]).to(device)
        # Add different weights to each label
        self.label_weights = torch.tensor([1.0, 1.0, 1.0, 1.0]).to(device)

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights,
                                           reduction='none')

    def forward(self, logits, targets):
        # logits shape: (batch_size, 3 * num_labels)
        num_labels = len(label_names)
        batch_size = logits.shape[0]
        logits = logits.reshape(batch_size * num_labels, 3)

        # targets shape: (batch_size, num_labels)
        targets = targets.to(torch.long)
        targets = targets.reshape(batch_size * num_labels)

        # weights
        label_weights = self.label_weights.repeat(
            batch_size, 1)  # Shape: (batch_size, num_labels)
        label_weights = label_weights.reshape(batch_size * num_labels)

        # weighted mean from class weights, same as reduction='mean'
        # https://pytorch.org/docs/main/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        class_weights_sum = self.class_weights[targets].sum()
        loss = self.ce_loss(logits, targets) / class_weights_sum

        # Apply weights to each different label
        loss = loss * label_weights
        return loss.sum()


class XGBoostClassifier:

    def __init__(self, num_classes):
        self.num_classes = num_classes
        xgb_model = xgb.XGBClassifier(objective='multi:softmax',
                                      eval_metric='mlogloss',
                                      num_class=3,
                                      use_label_encoder=False,
                                      verbosity=0)
        self.model = [xgb_model for _ in range(num_classes)]
        # Define hyperparameters for tuning
        self.param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200]
        }

    def fit(self, inputs, targets):
        if targets.shape[1] != self.num_classes:
            raise ValueError("Number of classes mismatch")

        # inputs shape: (batch_size, seq_len, input_dim)
        X_train = inputs[:, -1, :]  # Use only the last row for training

        # Perform grid search with cross-validation
        for i in range(self.num_classes):
            xgb_model = self.model[i]
            param_grid = self.param_grid
            y_class = targets[:, i]
            grid_search = GridSearchCV(xgb_model,
                                       param_grid,
                                       cv=5,
                                       scoring='accuracy',
                                       n_jobs=-1)
            grid_search.fit(X_train, y_class)

            # Train the best model
            self.model[i] = grid_search.best_estimator_

    def predict(self, inputs):

        X_test = inputs[:, -1, :]  # Use only the last row for prediction
        y_preds, y_probs = None, None
        for i in range(self.num_classes):
            xgb_model = self.model[i]
            y_pred_class = xgb_model.predict(X_test)
            y_pred_prob = xgb_model.predict_proba(X_test)
            if y_preds is None:
                y_preds, y_probs = y_pred_class, y_pred_prob
            else:
                y_preds = np.column_stack((y_preds, y_pred_class))
                y_probs = np.column_stack((y_probs, y_pred_prob))
        return y_preds, y_preds


def compute_model_output(model: PredictionModel, features: NDArray):
    if features is None or np.isnan(features).any() or np.isinf(
            features).any():
        # print(f"NaN or INF detected in {stock} on {date}")
        return None, None, None

    features_tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        logits = model(features_tensor)
        logits = logits.reshape(len(label_names), 3)
        probs = torch.softmax(
            logits, dim=1).float().numpy()  # convert logits to probabilities
        pred = calc_pred_labels(probs)

    return probs, pred, logits


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
