"""
Connect4Net — Primary Q-Network for Connect 4 DQN

Extracted verbatim from train_dqn_20251221.py (the validated architecture).
3 conv layers (64 channels each) + BN + Dropout2d, then FC 2688→128→7.

Architecture:
    Input: (batch, 2, 6, 7)  — canonical state (my_pieces, opp_pieces)
    ↓ Conv1: 2→64, 3×3, pad=1 + BN + ReLU + Dropout2d
    ↓ Conv2: 64→64, 3×3, pad=1 + BN + ReLU + Dropout2d
    ↓ Conv3: 64→64, 3×3, pad=1 + BN + ReLU + Dropout2d
    ↓ Flatten → 2688
    ↓ FC1: 2688→128 + ReLU + Dropout
    ↓ Output: 128→7  (raw Q-values, no activation)

He (Kaiming) weight init for ReLU nets.
Accepts np.ndarray or torch.Tensor; auto-inserts batch dim.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Connect4Net(nn.Module):
    """
    Convolutional Q-network for Connect 4 DQN.

    Args:
        device: Target device ("cpu", "mps", "cuda", or torch.device).
        dropout_rate: Dropout probability. Use 0.0 for evaluation instances;
                      0.1–0.2 during training for regularisation.

    Example::

        net = Connect4Net(device="cpu", dropout_rate=0.1)
        state = torch.randn(1, 2, 6, 7)
        q_values = net(state)           # shape (1, 7)
        action = q_values.argmax(1)     # greedy column choice
    """

    def __init__(
        self,
        device: "str | torch.device" = "cpu",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dropout_rate = dropout_rate

        # ---- Convolutional blocks (same as train_dqn_20251221.py) -------
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dr1 = nn.Dropout2d(p=dropout_rate)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dr2 = nn.Dropout2d(p=dropout_rate)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.dr3 = nn.Dropout2d(p=dropout_rate)

        # ---- Fully connected head ----------------------------------------
        # 64 filters × 6 rows × 7 cols = 2688
        self.fc1 = nn.Linear(64 * 6 * 7, 128)
        self.dr_fc = nn.Dropout(p=dropout_rate)
        self.output = nn.Linear(128, 7)

        # ---- He (Kaiming) weight initialisation --------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.to(self.device)

    # ------------------------------------------------------------------
    def forward(self, x: "np.ndarray | torch.Tensor") -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, 2, 6, 7) or (2, 6, 7). np.ndarray or torch.Tensor.

        Returns:
            (B, 7) raw Q-values — one per column.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(self.device)

        if x.dim() == 3:          # single state → add batch dim
            x = x.unsqueeze(0)

        x = F.relu(self.bn1(self.conv1(x)));  x = self.dr1(x)
        x = F.relu(self.bn2(self.conv2(x)));  x = self.dr2(x)
        x = F.relu(self.bn3(self.conv3(x)));  x = self.dr3(x)

        x = x.view(x.size(0), -1)   # flatten

        x = F.relu(self.fc1(x));    x = self.dr_fc(x)
        return self.output(x)       # raw Q-values

    # ------------------------------------------------------------------
    def get_config(self) -> dict:
        return {
            "dropout_rate": self.dropout_rate,
            "device": str(self.device),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(
                p.numel() for p in self.parameters() if p.requires_grad
            ),
        }

    def __repr__(self) -> str:
        cfg = self.get_config()
        return (
            f"Connect4Net("
            f"params={cfg['total_params']:,}, "
            f"dropout={cfg['dropout_rate']}, "
            f"device={cfg['device']})"
        )
