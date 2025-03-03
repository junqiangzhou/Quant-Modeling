import torch
from enum import Enum


class EncoderType(Enum):
    MLP = 0
    Transformer = 1
    LatentQueryTransformer = 2
    DualAttentionTransformer = 3


ENCODER_TYPE = EncoderType.Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
