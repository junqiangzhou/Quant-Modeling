import torch
import numpy as np

from model.model import PredictionModel

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

        self.price_model = PredictionModel(feature_len=len(feature_names),
                                           seq_len=look_back_window,
                                           encoder_type=ENCODER_TYPE)
        self.price_model.load_state_dict(
            torch.load(f"./model/export/{price_model_name}.pth"))
        self.price_model.eval()

    def predict(self, features):
        # Predict trend and price labels
        with torch.no_grad():
            trend_logits = self.trend_model(features)
            price_logits = self.price_model(features)

            trend_logits = trend_logits.reshape(len(label_names), 3)
            price_logits = price_logits.reshape(len(label_names), 3)

            trend_probs = torch.softmax(trend_logits,
                                        dim=1).float().cpu().numpy()
            price_probs = torch.softmax(price_logits,
                                        dim=1).float().cpu().numpy()

            # combine the two models' predictions
            probs = np.concatenate((trend_probs, price_probs), axis=0)

        return probs
