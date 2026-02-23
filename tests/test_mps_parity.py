"""
MPS (Apple Silicon) vs CPU Parity Test
=======================================

Checks that Connect4Net produces numerically consistent results on Apple's
Metal Performance Shaders (MPS) GPU vs CPU.

Why this matters
----------------
Apple's MPS backend uses float32 but different GPU kernels for BatchNorm,
matmul, and scatter operations.  Results are not bit-for-bit identical to
CPU but should be numerically close (within ~1e-3).  The failure mode we
guard against:
  - NaN/Inf outputs on MPS (silent kernel bugs)
  - Wildly different Q-values that would make MPS training diverge
  - Gradient explosion on MPS that doesn't occur on CPU

All tests in this file are skipped automatically if MPS is not available
(i.e., on non-Apple-Silicon machines or older macOS versions).

Numerical tolerance
-------------------
We use atol=5e-3 (0.005) for forward-pass comparison.  MPS BatchNorm
running stats and reduction kernels can differ from CPU by ~1e-4 to ~1e-3.
A 5e-3 tolerance accommodates this while catching large divergences.
"""

import copy
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from src.networks.connect4_net import Connect4Net
from src.environment.config import Config
from notebooks.training_examples_last_2_moves_20251221 import (
    generate_artificial_replay_buffer_for_training,
)


# ---------------------------------------------------------------------------
# Skip entire module if MPS is not available
# ---------------------------------------------------------------------------

MPS_AVAILABLE = (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)

pytestmark = pytest.mark.skipif(
    not MPS_AVAILABLE,
    reason="MPS not available on this machine — skipping Apple Silicon parity tests",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_net(device: str, seed: int = 42) -> Connect4Net:
    torch.manual_seed(seed)
    return Connect4Net(device=device, dropout_rate=0.0)


def _net_cpu_to_mps(net_cpu: Connect4Net) -> Connect4Net:
    """Copy an existing CPU net to MPS with identical weights."""
    net_mps = Connect4Net(device="mps", dropout_rate=0.0)
    net_mps.load_state_dict(net_cpu.state_dict())
    return net_mps


# ---------------------------------------------------------------------------
# 1. Device availability
# ---------------------------------------------------------------------------

class TestMPSAvailability:

    def test_mps_available(self):
        """MPS must be reported as available for these tests to be meaningful."""
        assert torch.backends.mps.is_available(), (
            "MPS not available — these tests should have been skipped"
        )

    def test_tensor_can_be_placed_on_mps(self):
        """Basic sanity: a tensor can live on MPS without error."""
        t = torch.zeros(2, 6, 7, device="mps")
        assert t.device.type == "mps"

    def test_connect4net_loads_on_mps(self):
        """Connect4Net can be instantiated on MPS without exception."""
        net = Connect4Net(device="mps", dropout_rate=0.0)
        assert next(net.parameters()).device.type == "mps"


# ---------------------------------------------------------------------------
# 2. Forward-pass parity: CPU vs MPS
# ---------------------------------------------------------------------------

class TestForwardPassParity:
    """Same weights + same input → outputs must be numerically close."""

    @pytest.fixture(scope="class")
    def nets_and_input(self):
        """Returns (net_cpu, net_mps, input_cpu, input_mps) with identical weights."""
        net_cpu = _make_net("cpu", seed=0)
        net_cpu.eval()
        net_mps = _net_cpu_to_mps(net_cpu)
        net_mps.eval()

        torch.manual_seed(1)
        x_cpu = torch.randn(4, 2, 6, 7)          # batch of 4 states
        x_mps = x_cpu.to("mps")
        return net_cpu, net_mps, x_cpu, x_mps

    def test_output_shape_same_on_mps(self, nets_and_input):
        """MPS output shape must match CPU output shape."""
        net_cpu, net_mps, x_cpu, x_mps = nets_and_input
        with torch.no_grad():
            out_cpu = net_cpu(x_cpu)
            out_mps = net_mps(x_mps)
        assert out_cpu.shape == out_mps.cpu().shape

    def test_output_values_close_cpu_vs_mps(self, nets_and_input):
        """MPS and CPU Q-values must agree within 5e-3 (absolute)."""
        net_cpu, net_mps, x_cpu, x_mps = nets_and_input
        with torch.no_grad():
            out_cpu = net_cpu(x_cpu).numpy()
            out_mps = net_mps(x_mps).cpu().numpy()
        max_diff = float(np.abs(out_cpu - out_mps).max())
        assert max_diff < 5e-3, (
            f"Max |CPU - MPS| = {max_diff:.6f} > 5e-3. "
            f"CPU sample: {out_cpu[0]}, MPS sample: {out_mps[0]}"
        )

    def test_no_nan_on_mps_forward(self, nets_and_input):
        """MPS forward pass must not produce NaN."""
        _, net_mps, _, x_mps = nets_and_input
        with torch.no_grad():
            out_mps = net_mps(x_mps).cpu().numpy()
        assert not np.any(np.isnan(out_mps)), "NaN in MPS forward pass"

    def test_no_inf_on_mps_forward(self, nets_and_input):
        """MPS forward pass must not produce Inf."""
        _, net_mps, _, x_mps = nets_and_input
        with torch.no_grad():
            out_mps = net_mps(x_mps).cpu().numpy()
        assert not np.any(np.isinf(out_mps)), "Inf in MPS forward pass"

    def test_argmax_agrees_cpu_vs_mps(self, nets_and_input):
        """Greedy action selection (argmax) must match on CPU and MPS.

        If Q-values are close, argmax should agree for all 4 batch entries.
        A mismatch would mean the agent picks different moves depending on
        device, which would cause inconsistency between training and eval.
        """
        net_cpu, net_mps, x_cpu, x_mps = nets_and_input
        with torch.no_grad():
            actions_cpu = net_cpu(x_cpu).argmax(1).numpy()
            actions_mps = net_mps(x_mps).argmax(1).cpu().numpy()
        mismatches = int(np.sum(actions_cpu != actions_mps))
        assert mismatches == 0, (
            f"{mismatches}/4 greedy actions differ between CPU and MPS. "
            f"CPU actions: {actions_cpu}, MPS actions: {actions_mps}"
        )


# ---------------------------------------------------------------------------
# 3. Weight parity after deepcopy / state_dict round-trip
# ---------------------------------------------------------------------------

class TestWeightTransfer:
    """Weights must survive CPU→MPS transfer without corruption."""

    def test_state_dict_round_trip(self):
        """Load CPU weights into MPS net; all parameters must match."""
        net_cpu = _make_net("cpu", seed=7)
        net_mps = _net_cpu_to_mps(net_cpu)

        cpu_state = net_cpu.state_dict()
        mps_state = net_mps.state_dict()

        for key in cpu_state:
            cpu_val = cpu_state[key].cpu().numpy()
            mps_val = mps_state[key].cpu().numpy()
            max_diff = float(np.abs(cpu_val - mps_val).max())
            assert max_diff < 1e-6, (
                f"Weight mismatch for '{key}': max_diff={max_diff:.2e}"
            )

    def test_no_nan_in_mps_weights(self):
        """Transferred weights must not contain NaN on MPS."""
        net_cpu = _make_net("cpu", seed=8)
        net_mps = _net_cpu_to_mps(net_cpu)
        for name, param in net_mps.named_parameters():
            vals = param.detach().cpu().numpy()
            assert not np.any(np.isnan(vals)), f"NaN in MPS param '{name}'"


# ---------------------------------------------------------------------------
# 4. Gradient and training step parity
# ---------------------------------------------------------------------------

class TestTrainingStepParity:
    """A single gradient step on CPU and MPS must produce similar results."""

    @pytest.fixture(scope="class")
    def one_step_results(self):
        """
        Run one identical training step on CPU and MPS from the same weights
        and same batch.  Returns (loss_cpu, loss_mps).
        """
        config = Config()
        replay_buffer = generate_artificial_replay_buffer_for_training()

        # Identical networks
        net_cpu = _make_net("cpu", seed=99)
        net_cpu.eval()
        net_mps = _net_cpu_to_mps(net_cpu)
        net_mps.eval()

        tgt_cpu = copy.deepcopy(net_cpu)
        tgt_mps = _net_cpu_to_mps(tgt_cpu)
        tgt_cpu.eval()
        tgt_mps.eval()

        opt_cpu = torch.optim.Adam(net_cpu.parameters(), lr=1e-3)
        opt_mps = torch.optim.Adam(net_mps.parameters(), lr=1e-3)

        # Same batch
        states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(
            16, indices=list(range(16))   # deterministic: always first 16 entries
        )

        def run_step(policy_net, target_net, optimizer, device):
            s  = torch.tensor(states,      dtype=torch.float32, device=device)
            a  = torch.tensor(actions,     dtype=torch.long,    device=device)
            r  = torch.tensor(rewards,     dtype=torch.float32, device=device)
            ns = torch.tensor(next_states, dtype=torch.float32, device=device)
            d  = torch.tensor(dones,       dtype=torch.float32, device=device)
            m  = torch.tensor(next_masks,  dtype=torch.float32, device=device)
            with torch.no_grad():
                nq = target_net(ns)
                nq.masked_fill_(m == 0, -1e9)
                tq = r - config.GAMMA * nq.max(1)[0] * (1 - d)
            policy_net.train()
            optimizer.zero_grad()
            qv = policy_net(s)
            pq = qv.gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(pq, tq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
            optimizer.step()
            return loss.item()

        loss_cpu = run_step(net_cpu, tgt_cpu, opt_cpu, "cpu")
        loss_mps = run_step(net_mps, tgt_mps, opt_mps, "mps")
        return loss_cpu, loss_mps

    def test_loss_finite_on_cpu(self, one_step_results):
        loss_cpu, _ = one_step_results
        assert np.isfinite(loss_cpu), f"CPU loss not finite: {loss_cpu}"

    def test_loss_finite_on_mps(self, one_step_results):
        _, loss_mps = one_step_results
        assert np.isfinite(loss_mps), f"MPS loss not finite: {loss_mps}"

    def test_loss_values_close_cpu_vs_mps(self, one_step_results):
        """CPU and MPS losses must agree within 1% of the CPU loss value."""
        loss_cpu, loss_mps = one_step_results
        rel_diff = abs(loss_cpu - loss_mps) / (abs(loss_cpu) + 1e-8)
        assert rel_diff < 0.01, (
            f"Loss relative difference {rel_diff:.4f} > 1%. "
            f"CPU={loss_cpu:.6f}, MPS={loss_mps:.6f}"
        )


# ---------------------------------------------------------------------------
# 5. Runtime comparison: CPU vs MPS
# ---------------------------------------------------------------------------

class TestRuntimeComparison:
    """Measure and report forward-pass throughput on CPU vs MPS.

    We do NOT assert MPS is faster — for small mini-batches, MPS kernel
    launch overhead can make it *slower* than CPU.  For training-scale
    batches (128) it should be faster.  The test asserts:
      1. MPS completes in finite time (no hang/crash)
      2. Throughput is reported so the developer can make an informed choice
      3. For large batch (128), MPS should not be more than 5× *slower* than CPU
         (if MPS were that slow there'd be a driver issue)
    """

    def test_forward_throughput_comparison(self):
        """Time 500 forward passes at batch=128 on CPU and MPS, print results."""
        import time

        REPS = 500
        BATCH = 128
        torch.manual_seed(42)
        x = torch.randn(BATCH, 2, 6, 7)

        net_cpu = _make_net("cpu", seed=42); net_cpu.eval()
        net_mps = _net_cpu_to_mps(net_cpu); net_mps.eval()
        x_mps = x.to("mps")

        # Warm-up (excludes JIT / kernel compilation latency)
        with torch.no_grad():
            for _ in range(10):
                net_cpu(x)
            for _ in range(10):
                net_mps(x_mps)
            # Flush MPS command queue
            torch.mps.synchronize()

        # CPU timing
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(REPS):
                net_cpu(x)
        cpu_sec = time.perf_counter() - t0

        # MPS timing (synchronize after to ensure all GPU work is done)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(REPS):
                net_mps(x_mps)
        torch.mps.synchronize()
        mps_sec = time.perf_counter() - t0

        speedup = cpu_sec / mps_sec
        cpu_fps  = REPS * BATCH / cpu_sec
        mps_fps  = REPS * BATCH / mps_sec

        print(
            f"\n  [runtime] batch={BATCH}, reps={REPS}\n"
            f"  CPU : {cpu_sec:.3f}s  ({cpu_fps:,.0f} states/s)\n"
            f"  MPS : {mps_sec:.3f}s  ({mps_fps:,.0f} states/s)\n"
            f"  Speedup: {speedup:.2f}×  ({'MPS faster' if speedup > 1 else 'CPU faster'})"
        )

        # MPS must complete in finite time and not be catastrophically slow
        assert mps_sec > 0, "MPS timing was zero — something went wrong"
        assert np.isfinite(mps_sec), "MPS timing is non-finite"
        # MPS must not be more than 10× slower than CPU (would indicate driver issue)
        assert speedup > 0.1, (
            f"MPS is {1/speedup:.1f}× slower than CPU — potential MPS driver issue"
        )

    def test_float32_drift_many_inputs(self):
        """Check float32 drift across 200 random inputs.

        For each input, compute |CPU_out - MPS_out| for all 7 Q-values.
        Record the max absolute difference across the entire test set.
        Must stay < 2e-2 (20× our 1e-3 expected drift).

        This is the key rounding safety check — if MPS has a systematic
        float32 precision issue, the max drift would climb above this.
        """
        N = 200
        torch.manual_seed(123)
        net_cpu = _make_net("cpu", seed=123); net_cpu.eval()
        net_mps = _net_cpu_to_mps(net_cpu); net_mps.eval()

        max_drift = 0.0
        drifts = []
        with torch.no_grad():
            for _ in range(N):
                x = torch.randn(1, 2, 6, 7)
                out_cpu = net_cpu(x).numpy()
                out_mps = net_mps(x.to("mps")).cpu().numpy()
                drift = float(np.abs(out_cpu - out_mps).max())
                drifts.append(drift)
                max_drift = max(max_drift, drift)

        mean_drift = float(np.mean(drifts))
        p99_drift  = float(np.percentile(drifts, 99))
        print(
            f"\n  [float32 drift] N={N} random inputs\n"
            f"  max_drift  = {max_drift:.2e}\n"
            f"  mean_drift = {mean_drift:.2e}\n"
            f"  p99_drift  = {p99_drift:.2e}"
        )

        assert max_drift < 2e-2, (
            f"Float32 drift max={max_drift:.2e} > 2e-2. "
            f"MPS may have systematic precision issues."
        )
        assert not np.any(np.isnan(drifts)), "NaN drift detected — MPS producing NaN"


# ---------------------------------------------------------------------------
# 6. Short training convergence on MPS
# ---------------------------------------------------------------------------

class TestMPSConvergence:
    """MPS training must converge on the synthetic buffer, same as CPU."""

    def test_mps_synthetic_learning_converges(self):
        """200 steps on MPS → 100% win Q-values positive, loss decreases.

        Uses the same synthetic buffer and asserts as test_synthetic_learning.py
        but runs on MPS device.
        """
        config = Config()
        replay_buffer = generate_artificial_replay_buffer_for_training()
        wins = [(e.state, e.action) for e in replay_buffer.buffer
                if float(e.done) == 1.0 and float(e.reward) > 0.5]

        net = Connect4Net(device="mps", dropout_rate=0.0)
        tgt = copy.deepcopy(net); tgt.eval(); net.eval()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)

        loss_history = []
        for step in range(200):
            states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(16)
            s  = torch.tensor(states,      dtype=torch.float32, device="mps")
            a  = torch.tensor(actions,     dtype=torch.long,    device="mps")
            r  = torch.tensor(rewards,     dtype=torch.float32, device="mps")
            ns = torch.tensor(next_states, dtype=torch.float32, device="mps")
            d  = torch.tensor(dones,       dtype=torch.float32, device="mps")
            m  = torch.tensor(next_masks,  dtype=torch.float32, device="mps")
            with torch.no_grad():
                nq = tgt(ns); nq.masked_fill_(m == 0, -1e9)
                tq = r - config.GAMMA * nq.max(1)[0] * (1 - d)
            opt.zero_grad()
            qv = net(s)
            pq = qv.gather(1, a.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(pq, tq)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            loss_history.append(loss.item())
            if (step + 1) % 50 == 0:
                tgt.load_state_dict(net.state_dict())

        # Loss must be finite throughout
        assert all(np.isfinite(l) for l in loss_history), "Non-finite loss on MPS"

        # Loss must decrease
        early = np.mean(loss_history[:20])
        late  = np.mean(loss_history[-20:])
        assert late < early, (
            f"Loss did not decrease on MPS: early={early:.4f}, late={late:.4f}"
        )

        # Win Q-values must be positive (100%)
        net.eval()
        q_vals = []
        with torch.no_grad():
            for state, action in wins:
                st = torch.tensor(np.array(state), dtype=torch.float32,
                                  device="mps").unsqueeze(0)
                q_vals.append(net(st)[0, action].item())
        failures = [q for q in q_vals if q <= 0.0]
        assert len(failures) == 0, (
            f"{len(failures)}/{len(wins)} win Q-values ≤ 0 on MPS"
        )
