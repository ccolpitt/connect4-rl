"""
Unit tests for Connect4Net (src/networks/connect4_net.py).

Covers:
  - Output shape for single and batched inputs (tensor + numpy)
  - NaN/Inf guard on random/zero/full inputs
  - He weight init: conv/fc non-zero, biases zero, BN weight=1
  - Deep-copy → identical output; gradient step → divergence
  - load_state_dict syncs policy→target
  - Eval mode deterministic; train+dropout stochastic
  - Gradients reach all parameters; loss decreases over 100 steps
  - get_config() keys and param-count sanity
"""

import copy
import pytest
import numpy as np
import torch

from networks.connect4_net import Connect4Net


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def net():
    m = Connect4Net(device="cpu", dropout_rate=0.0)
    m.eval()
    return m


@pytest.fixture
def net_dropout():
    return Connect4Net(device="cpu", dropout_rate=0.2)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:

    def test_single_tensor_shape(self, net):
        out = net(torch.zeros(2, 6, 7))
        assert out.shape == (1, 7)

    @pytest.mark.parametrize("B", [1, 4, 16, 32])
    def test_batched_tensor_shape(self, net, B):
        out = net(torch.randn(B, 2, 6, 7))
        assert out.shape == (B, 7)

    def test_numpy_single_shape(self, net):
        out = net(np.zeros((2, 6, 7), dtype=np.float32))
        assert out.shape == (1, 7)

    def test_numpy_batch_shape(self, net):
        out = net(np.zeros((8, 2, 6, 7), dtype=np.float32))
        assert out.shape == (8, 7)


# ---------------------------------------------------------------------------
# 2. Numerical validity
# ---------------------------------------------------------------------------

class TestNumericalValidity:

    @pytest.mark.parametrize("input_fn", [
        lambda: torch.randn(16, 2, 6, 7),
        lambda: torch.zeros(1, 2, 6, 7),
        lambda: torch.ones(1, 2, 6, 7),
    ])
    def test_no_nan_inf(self, net, input_fn):
        with torch.no_grad():
            out = net(input_fn())
        assert not torch.isnan(out).any(), "NaN in Q-values"
        assert not torch.isinf(out).any(), "Inf in Q-values"


# ---------------------------------------------------------------------------
# 3. Weight initialisation
# ---------------------------------------------------------------------------

class TestWeightInit:

    def test_conv_weights_nonzero(self):
        net = Connect4Net(device="cpu")
        for name, p in net.named_parameters():
            if "conv" in name and "weight" in name:
                assert p.abs().sum().item() > 0, f"{name} all zeros"

    def test_all_biases_zero(self):
        net = Connect4Net(device="cpu")
        for name, p in net.named_parameters():
            if "bias" in name:
                assert torch.allclose(p, torch.zeros_like(p)), \
                    f"{name} bias not zero: {p.abs().max():.6f}"

    def test_bn_weight_one(self):
        net = Connect4Net(device="cpu")
        for name, p in net.named_parameters():
            if "bn" in name and "weight" in name:
                assert torch.allclose(p, torch.ones_like(p)), \
                    f"{name} BN weight not 1"


# ---------------------------------------------------------------------------
# 4. Copy / independence
# ---------------------------------------------------------------------------

class TestCopyIndependence:

    def test_deepcopy_identical_output(self):
        net = Connect4Net(device="cpu")
        clone = copy.deepcopy(net)
        state = torch.randn(4, 2, 6, 7)
        net.eval(); clone.eval()
        with torch.no_grad():
            assert torch.allclose(net(state), clone(state))

    def test_two_fresh_nets_differ(self):
        # With different random seeds they should differ (no seed set)
        net1 = Connect4Net(device="cpu")
        net2 = Connect4Net(device="cpu")
        state = torch.randn(1, 2, 6, 7)
        net1.eval(); net2.eval()
        with torch.no_grad():
            out1, out2 = net1(state), net2(state)
        # This will pass unless seeds are perfectly correlated
        # (astronomically unlikely with He init)
        assert not torch.allclose(out1, out2, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Target-net sync
# ---------------------------------------------------------------------------

class TestTargetNetSync:

    def test_load_state_dict_syncs(self):
        policy = Connect4Net(device="cpu")
        target = Connect4Net(device="cpu")
        _ = policy(torch.randn(4, 2, 6, 7))   # warm up BN running stats
        target.load_state_dict(policy.state_dict())

        state = torch.randn(4, 2, 6, 7)
        policy.eval(); target.eval()
        with torch.no_grad():
            assert torch.allclose(policy(state), target(state))

    def test_gradient_step_diverges_from_frozen_target(self):
        policy = Connect4Net(device="cpu")
        target = Connect4Net(device="cpu")
        target.load_state_dict(policy.state_dict())

        opt = torch.optim.Adam(policy.parameters(), lr=0.01)
        dummy = torch.randn(4, 2, 6, 7)
        policy(dummy).sum().backward()
        opt.step()

        state = torch.randn(4, 2, 6, 7)
        policy.eval(); target.eval()
        with torch.no_grad():
            assert not torch.allclose(policy(state), target(state), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Eval / train mode
# ---------------------------------------------------------------------------

class TestEvalTrainMode:

    def test_eval_deterministic(self, net_dropout):
        net_dropout.eval()
        state = torch.randn(1, 2, 6, 7)
        with torch.no_grad():
            assert torch.allclose(net_dropout(state), net_dropout(state))

    def test_train_dropout_stochastic(self, net_dropout):
        net_dropout.train()
        state = torch.randn(1, 2, 6, 7)
        outputs = []
        for _ in range(10):
            with torch.no_grad():
                outputs.append(net_dropout(state).squeeze().tolist())
        unique = {tuple(round(v, 5) for v in o) for o in outputs}
        assert len(unique) > 1, "Train+dropout should be stochastic"


# ---------------------------------------------------------------------------
# 7. Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:

    def test_gradients_reach_all_params(self):
        net = Connect4Net(device="cpu", dropout_rate=0.0)
        net.train()
        state = torch.randn(4, 2, 6, 7)
        ((net(state) - torch.zeros(4, 7)) ** 2).mean().backward()
        for name, p in net.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad: {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad: {name}"

    def test_loss_decreases_100_steps(self):
        torch.manual_seed(0)
        net = Connect4Net(device="cpu", dropout_rate=0.0)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        state = torch.randn(16, 2, 6, 7)
        target = torch.zeros(16, 7)
        losses = []
        for _ in range(100):
            opt.zero_grad()
            loss = ((net(state) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], \
            f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"


# ---------------------------------------------------------------------------
# 8. Config / repr
# ---------------------------------------------------------------------------

class TestConfig:

    def test_get_config_has_required_keys(self, net):
        cfg = net.get_config()
        for key in ("dropout_rate", "device", "total_params", "trainable_params"):
            assert key in cfg

    def test_param_count_sane(self, net):
        # 3×(conv 3×3×2×64 + bias + BN) + FC layers ≈ 450k–600k params
        cfg = net.get_config()
        assert 100_000 < cfg["total_params"] < 5_000_000, \
            f"Unexpected param count: {cfg['total_params']}"

    def test_repr_contains_connect4net(self, net):
        assert "Connect4Net" in repr(net)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
