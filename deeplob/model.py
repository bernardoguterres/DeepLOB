"""DeepLOB model architecture.

CNN feature extractor → Inception multi-scale block → LSTM → linear classifier.

Reference:
    Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional
    Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing,
    67(11), 3001–3012.  https://arxiv.org/abs/1902.09450
"""

import torch
import torch.nn as nn

__all__ = ["CNNBlock", "InceptionModule", "DeepLOB", "count_parameters"]

#: CNNBlock's output shape for the fixed (100, 40) input window: (channels, time, levels).
CNN_OUT_CHANNELS = 32
CNN_OUT_HEIGHT = 94
CNN_OUT_WIDTH = 20


class CNNBlock(nn.Module):
    """First block of DeepLOB: spatial feature extraction from LOB levels.

    Applies three Conv2d layers. The first kernel (1×2) pairs each price
    with its volume at the same LOB level — the key spatial insight from
    Zhang et al. (2019). Subsequent (4×1) kernels capture short-range
    temporal patterns.

    Input shape:  ``(batch, 1, 100, 40)``
    Output shape: ``(batch, 32, 94, 20)``
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Pair each price with its volume: 40 columns → 20 (stride=2 in W)
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            # Short-range temporal: 100 rows → 97
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            # Short-range temporal: 97 rows → 94
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply three Conv2d layers with LeakyReLU + BatchNorm activations.

        Args:
            x: Input tensor of shape ``(batch, 1, 100, 40)``.

        Returns:
            Feature tensor of shape ``(batch, 32, 94, 20)``.
        """
        return self.layers(x)


class InceptionModule(nn.Module):
    """Inception-style block capturing multi-scale temporal patterns.

    Runs three parallel branches with different temporal kernel sizes,
    then concatenates along the channel dimension. This allows the model
    to simultaneously capture fast micro-structure (1-event) and slower
    patterns (3–5 events) without manually selecting a single timescale.

    Input shape:  ``(batch, 32, T, 20)``
    Output shape: ``(batch, 192, T, 20)`` — 64 channels per branch × 3
    """

    def __init__(self) -> None:
        super().__init__()

        # Branch A: 1×1 bottleneck, then 3×1 temporal convolution (same-size)
        self.branch_a = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # Branch B: 1×1 bottleneck, then 5×1 temporal convolution (same-size)
        self.branch_b = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding=(2, 0)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

        # Branch C: max-pool (same-size) then 1×1 channel projection
        self.branch_c = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three branches in parallel and concatenate on the channel axis.

        Args:
            x: Input tensor of shape ``(batch, 32, T, 20)``.

        Returns:
            Feature tensor of shape ``(batch, 192, T, 20)``.
        """
        return torch.cat([self.branch_a(x), self.branch_b(x), self.branch_c(x)], dim=1)


class DeepLOB(nn.Module):
    """Full DeepLOB architecture: CNN → Inception → LSTM → classifier.

    Reimplementation of Zhang et al. (2019), "DeepLOB: Deep Convolutional
    Neural Networks for Limit Order Books."

    Args:
        hidden_size: LSTM hidden state size (default 256).
        num_lstm_layers: Number of LSTM layers (default 1).
        num_classes: Number of output classes (default 3).

    Input shape:  ``(batch, 1, 100, 40)``
    Output shape: ``(batch, 3)`` — raw logits, no softmax applied
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_lstm_layers: int = 1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.cnn = CNNBlock()
        self.inception = InceptionModule()
        # After CNN+Inception: (B, 192, 94, 20); flatten spatial → seq length 1880, features 192
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full forward pass.

        Args:
            x: LOB input tensor of shape ``(batch, 1, 100, 40)``.

        Returns:
            Class logits of shape ``(batch, num_classes)``. Apply softmax
            (e.g. via :func:`torch.nn.functional.softmax`) to get probabilities.
        """
        B = x.shape[0]
        x = self.cnn(x)  # (B, 32, 94, 20)
        x = self.inception(x)  # (B, 192, 94, 20)
        x = x.permute(0, 2, 3, 1)  # (B, 94, 20, 192) — time × space × channels
        x = x.reshape(
            B, CNN_OUT_HEIGHT * CNN_OUT_WIDTH, 192
        )  # (B, 1880, 192) — flatten spatial into sequence
        x, _ = self.lstm(x)  # (B, 1880, hidden_size)
        x = x[:, -1, :]  # (B, hidden_size) — final timestep only
        x = self.fc(x)  # (B, num_classes)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of trainable parameters (``requires_grad=True`` only).
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
