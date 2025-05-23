import torch
import numpy as np
import os
from numpy.typing import NDArray

from model.model import PredictionModel, compute_model_output

from config.config import (ENCODER_TYPE, label_names, feature_names,
                           look_back_window, MODEL_EXPORT_NAME, LabelType,
                           LABEL_TYPE)


# This provies a wrapper that combines the two models and predicts the output of both trend and price labels
class JointLabelPredictor:

    def __init__(self):
        self.trend_model = None
        self.price_model = None

        self._load_model(MODEL_EXPORT_NAME)

    def _load_model(self, model_name):
        # Load model
        if LABEL_TYPE == LabelType.PRICE:
            trend_model_name = model_name.replace("price", "trend")
            price_model_name = model_name
        else:
            trend_model_name = model_name
            price_model_name = model_name.replace("trend", "price")

        self.trend_model = PredictionModel(feature_len=len(feature_names),
                                           seq_len=look_back_window,
                                           encoder_type=ENCODER_TYPE)
        self.trend_model.load_state_dict(
            torch.load(f"./model/export/{trend_model_name}.pth"))
        self.trend_model.eval()

        if os.path.exists(f"./model/export/{price_model_name}.pth"):
            self.price_model = PredictionModel(feature_len=len(feature_names),
                                               seq_len=look_back_window,
                                               encoder_type=ENCODER_TYPE)
            self.price_model.load_state_dict(
                torch.load(f"./model/export/{price_model_name}.pth"))
            self.price_model.eval()

    def predict(self, features: NDArray) -> NDArray:
        # Predict trend and price labels
        trend_probs, _, _ = compute_model_output(self.trend_model, features)
        if self.price_model is not None:
            price_probs, _, _ = compute_model_output(self.price_model,
                                                     features)
        else:
            price_probs = np.zeros_like(trend_probs)

        # combine the two models' predictions
        probs = np.concatenate((trend_probs, price_probs), axis=0)

        return probs
