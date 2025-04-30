from model.model import CustomLoss
from config.config import device

import torch
import torch.nn as nn
import pytest

batch_size = 2
num_labels = 4
num_classes = 3


def test_custom_loss_shapes():

    loss_fn = CustomLoss().to(device)
    logits = torch.randn(batch_size, num_labels * num_classes).to(
        device)  # Shape: (batch_size, num_labels * num_classes)
    targets = torch.randint(0, num_classes, (batch_size, num_labels)).to(
        device)  # Shape: (batch_size, num_labels)

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, "Loss should be a scalar value"
    assert loss.item() >= 0, "Loss should be non-negative"


def test_custom_loss_reduction():
    loss_fn = CustomLoss().to(device)

    logits = torch.tensor([[
        2.0,
        1.0,
        0.1,
        1.5,
        2.0,
        1.0,
        0.1,
        1.5,
        2.0,
        1.0,
        0.1,
        1.5,
    ], [0.5, 1.2, 2.1, 0.5, 1.2, 2.1, 0.5, 1.2, 2.1, 0.5, 1.2,
        2.1]]).to(device)  # Shape: (2, num_labels * num_classes)
    targets = torch.tensor([[0, 2, 1, 0],
                            [2, 1, 1, 1]]).to(device)  # Shape: (2, num_labels)

    loss = loss_fn(logits, targets)
    print(loss)

    assert torch.isfinite(loss).all(), "Loss should be finite"


@pytest.mark.skip
def test_custom_loss_gradients():
    logits = torch.randn(batch_size,
                         num_labels * num_classes,
                         requires_grad=True).to(device)
    targets = torch.randint(0, num_classes,
                            (batch_size, num_labels)).to(device)

    loss_fn = CustomLoss().to(device)
    loss = loss_fn(logits, targets)
    loss.backward()

    assert logits.grad is not None, "Gradients should be computed"
    assert torch.isfinite(logits.grad).all(), "Gradients should be finite"


def test_custom_loss_different_weights():
    loss_fn = CustomLoss().to(device)

    logits = torch.randn(batch_size, num_labels * num_classes).to(device)
    targets = torch.randint(0, num_classes,
                            (batch_size, num_labels)).to(device)

    loss1 = loss_fn(logits, targets)

    loss_fn.label_weights = torch.tensor([0.5, 1.5, 1.0, 2.0
                                          ]).to(device)  # Change label weights
    loss2 = loss_fn(logits, targets)

    assert loss1.item() != loss2.item(
    ), "Loss should change when label weights are different"


if __name__ == "__main__":
    pytest.main(["-v", "test_custom_loss.py"])
