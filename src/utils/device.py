"""
Device Selection Utility
========================

Automatically selects the best available compute device.  Uses Apple MPS
(Metal Performance Shaders) if and only if all safety checks pass; otherwise
falls back to CPU silently.

The checks run once at module import time and are cached.

Safety checks performed before accepting MPS
---------------------------------------------
1. MPS is available (`torch.backends.mps.is_available()`)
2. A float32 tensor can be created on MPS without error
3. Connect4Net forward pass produces finite (non-NaN, non-Inf) output on MPS
4. CPU and MPS Q-value outputs agree within 5e-3 (float32 precision tolerance)
5. Backward pass on MPS produces finite gradients
   (uses synthetic random data — no Config/replay-buffer import to avoid
   circular dependencies)

If any check fails, MPS is rejected and CPU is used.

Usage
-----
    from src.utils.device import get_device

    device = get_device()          # "mps" or "cpu"
    net = Connect4Net(device=device)

    # Pre-computed at import time:
    from src.utils.device import DEVICE

Run `python src/utils/device.py` to see which device was selected and why.
The full parity test suite is in tests/test_mps_parity.py.
"""

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Tolerance for CPU vs MPS float32 comparison
# ──────────────────────────────────────────────────────────────────────────────
_MPS_ATOL = 5e-3   # absolute tolerance on Q-value outputs


def _mps_is_safe() -> tuple[bool, str]:
    """
    Run quick MPS safety checks.

    Returns
    -------
    (ok: bool, reason: str)
        ok=True  → MPS passed all checks, safe to use
        ok=False → reason explains why MPS was rejected
    """
    # ── Check 1: MPS available ──────────────────────────────────────────────
    if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return False, "MPS not available on this machine"

    try:
        # ── Check 2: Basic tensor creation ─────────────────────────────────
        t = torch.zeros(2, 6, 7, dtype=torch.float32, device="mps")
        if t.device.type != "mps":
            return False, "MPS tensor creation failed"

        # ── Check 3 & 4: Forward-pass parity ───────────────────────────────
        # Import inside function to avoid circular dependency:
        #   config.py -> device.py -> connect4_net.py  (no cycle)
        from src.networks.connect4_net import Connect4Net

        torch.manual_seed(0)
        net_cpu = Connect4Net(device="cpu", dropout_rate=0.0)
        net_cpu.eval()

        net_mps = Connect4Net(device="mps", dropout_rate=0.0)
        net_mps.load_state_dict(net_cpu.state_dict())
        net_mps.eval()

        torch.manual_seed(1)
        x_cpu = torch.randn(4, 2, 6, 7)
        x_mps = x_cpu.to("mps")

        with torch.no_grad():
            out_cpu = net_cpu(x_cpu).numpy()
            out_mps = net_mps(x_mps).cpu().numpy()

        # NaN / Inf check
        if not np.all(np.isfinite(out_mps)):
            return False, "MPS forward pass produced NaN or Inf"

        # Parity check
        max_diff = float(np.abs(out_cpu - out_mps).max())
        if max_diff > _MPS_ATOL:
            return False, (
                f"CPU/MPS Q-value drift {max_diff:.2e} > tolerance {_MPS_ATOL:.0e}. "
                f"Possible float32 precision issue on this MPS driver version."
            )

        # ── Check 5: Backward pass with finite gradients ────────────────────
        # Use synthetic random data — avoids importing Config or replay-buffer
        # (which would create a circular import: config → device → config).
        import copy
        import torch.nn as nn

        net_train = copy.deepcopy(net_mps)
        net_train.train()
        opt = torch.optim.Adam(net_train.parameters(), lr=1e-3)

        x_rand   = torch.randn(16, 2, 6, 7, device="mps")
        tgt_rand = torch.randn(16, device="mps")      # random regression targets

        opt.zero_grad()
        qv   = net_train(x_rand)
        pq   = qv[:, 3]                               # pretend action=3 for all
        loss = nn.functional.mse_loss(pq, tgt_rand)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_train.parameters(), 1.0)
        opt.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            return False, f"MPS backward pass produced non-finite loss: {loss_val}"

        for name, param in net_train.named_parameters():
            if param.grad is not None:
                g = param.grad.cpu().numpy()
                if not np.all(np.isfinite(g)):
                    return False, f"Non-finite gradient in MPS param '{name}'"

    except Exception as exc:
        return False, f"MPS check raised exception: {type(exc).__name__}: {exc}"

    return True, "all checks passed"


# ──────────────────────────────────────────────────────────────────────────────
# Run checks once at import time and cache result
# ──────────────────────────────────────────────────────────────────────────────

_mps_ok, _mps_reason = _mps_is_safe()

if _mps_ok:
    DEVICE: str = "mps"
    _device_source = "MPS (Apple Silicon GPU — all safety checks passed)"
else:
    DEVICE: str = "cpu"
    _device_source = f"CPU (MPS rejected: {_mps_reason})"


def get_device() -> str:
    """
    Return the best available compute device: "mps" or "cpu".

    MPS is only returned if all safety checks pass (parity, no NaN,
    finite gradients).  Otherwise falls back to "cpu".

    Returns
    -------
    str: "mps" or "cpu"
    """
    return DEVICE


def get_device_info() -> dict:
    """
    Return a dict with full device selection details.

    Useful for logging at the start of a training run.

    Returns
    -------
    dict with keys: device, mps_available, mps_accepted, reason, source
    """
    return {
        "device": DEVICE,
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "mps_accepted": _mps_ok,
        "reason": _mps_reason,
        "source": _device_source,
    }


if __name__ == "__main__":
    info = get_device_info()
    print(f"Selected device : {info['device']}")
    print(f"MPS available   : {info['mps_available']}")
    print(f"MPS accepted    : {info['mps_accepted']}")
    print(f"Reason          : {info['reason']}")
