"""
Phase 4 — Synthetic Replay Buffer Learning Test
================================================

Validates that Connect4Net can perfectly learn correct Q-values from the
hand-crafted replay buffer in notebooks/training_examples_last_2_moves_20251221.py.

The buffer contains 11 win/loss board positions stored with add_symmetric():
  - 11 examples × 2 moves (losing + winning) × 2 symmetric versions = 44 entries
  - 22 win-terminal transitions  (reward = +1.0, done = True)
  - 22 loss-terminal transitions (reward = -1.0, done = True, via update_penalty)

Why 100% correct signs is the right bar
-----------------------------------------
All 44 entries are terminal (done=True), so the Bellman target collapses to:

    target_q = reward          (bootstrap term vanishes: gamma * ... * (1-done) = 0)

There is zero bootstrap noise.  Every target is exactly +1.0 or -1.0.
The buffer is tiny, clean, and hand-crafted.  If the network cannot achieve
100% correct signs here, it will never work during noisy self-play.

Diagnostic result (from notebooks/diagnose_synthetic_learning.py):
  Steps=100: 22/22 wins correct, 22/22 losses correct, min win Q=+0.644
  Steps=500: 22/22 wins correct, 22/22 losses correct, min win Q=+0.926

Bellman target (NegaMax, matching train_dqn_20251221.py)
--------------------------------------------------------
    target_q = r  -  gamma * max(Q_target(s'))  * (1 - done)
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
# Helpers
# ---------------------------------------------------------------------------

def _collect_terminal_entries(replay_buffer, reward_sign: float):
    """Return list of (state, action) for all terminal entries matching reward_sign."""
    return [
        (e.state, e.action)
        for e in replay_buffer.buffer
        if float(e.done) == 1.0 and abs(float(e.reward) - reward_sign) < 1e-4
    ]


def _eval_q_for_entries(policy_net, entries):
    """Return list of Q(s, a_taken) for every (state, action) pair."""
    qs = []
    policy_net.eval()
    with torch.no_grad():
        for state, action in entries:
            s_t = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0)
            q = policy_net(s_t)[0, action].item()
            qs.append(q)
    return qs


# ---------------------------------------------------------------------------
# Module-scoped fixtures — built once per test session to save time
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_buffer():
    """Build the hand-crafted replay buffer once for the whole module."""
    return generate_artificial_replay_buffer_for_training()


@pytest.fixture(scope="module")
def trained_net_and_losses(synthetic_buffer):
    """
    Train Connect4Net on the synthetic buffer for 500 gradient steps.

    Configuration
    -------------
    - batch_size = 16   (buffer has 44 entries, fits comfortably)
    - Adam lr = 1e-3
    - Target-net sync every 50 steps
    - Both nets in eval() mode: BN stats frozen, dropout disabled
    - NegaMax Bellman: target = r - gamma * max(Q_target(s')) * (1 - done)
    - Gradient clipping at 1.0 (same as production loop)

    Returns
    -------
    (policy_net, loss_history)  — loss_history is a list of 500 floats.
    """
    config = Config()
    device = "cpu"

    policy_net = Connect4Net(device=device, dropout_rate=0.0)
    target_net = copy.deepcopy(policy_net)
    target_net.eval()
    policy_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)

    BATCH_SIZE = 16
    NUM_STEPS = 500
    TARGET_SYNC_FREQ = 50
    GAMMA = config.GAMMA

    loss_history: list[float] = []

    for step in range(NUM_STEPS):
        states, actions, rewards, next_states, dones, next_masks = synthetic_buffer.sample(
            BATCH_SIZE
        )

        s_batch  = torch.tensor(states,      dtype=torch.float32, device=device)
        a_batch  = torch.tensor(actions,     dtype=torch.long,    device=device)
        r_batch  = torch.tensor(rewards,     dtype=torch.float32, device=device)
        ns_batch = torch.tensor(next_states, dtype=torch.float32, device=device)
        d_batch  = torch.tensor(dones,       dtype=torch.float32, device=device)
        m_batch  = torch.tensor(next_masks,  dtype=torch.float32, device=device)

        with torch.no_grad():
            next_q = target_net(ns_batch)
            masked_next_q = next_q.masked_fill(m_batch == 0, -1e9)
            next_q_max = masked_next_q.max(dim=1)[0]
            target_q = r_batch - GAMMA * next_q_max * (1 - d_batch)

        optimizer.zero_grad()
        q_values = policy_net(s_batch)
        predicted_qs = q_values.gather(1, a_batch.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(predicted_qs, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        loss_history.append(loss.item())

        if (step + 1) % TARGET_SYNC_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return policy_net, loss_history


# ---------------------------------------------------------------------------
# 1. Buffer structure sanity checks (no training needed)
# ---------------------------------------------------------------------------

class TestSyntheticBufferStructure:
    """Verify the hand-crafted buffer is built correctly before training."""

    def test_buffer_has_expected_size(self, synthetic_buffer):
        """11 examples × 2 moves × 2 symmetric = 44 entries exactly."""
        assert len(synthetic_buffer) == 44, (
            f"Expected 44 entries (11 examples × 2 moves × 2 symmetric), "
            f"got {len(synthetic_buffer)}"
        )

    def test_buffer_has_22_win_terminals(self, synthetic_buffer):
        """Exactly 22 win-terminal entries (reward=+1.0, done=True)."""
        wins = _collect_terminal_entries(synthetic_buffer, +1.0)
        assert len(wins) == 22, (
            f"Expected 22 win terminal entries, got {len(wins)}"
        )

    def test_buffer_has_22_loss_terminals(self, synthetic_buffer):
        """Exactly 22 loss-terminal entries (reward=-1.0, done=True) from update_penalty."""
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        assert len(losses) == 22, (
            f"Expected 22 loss terminal entries, got {len(losses)}"
        )

    def test_all_states_are_2_channel(self, synthetic_buffer):
        """Every entry in the buffer must have state shape (2, 6, 7)."""
        for i, entry in enumerate(synthetic_buffer.buffer):
            shape = np.array(entry.state).shape
            assert shape == (2, 6, 7), (
                f"Entry {i}: expected shape (2,6,7), got {shape}"
            )

    def test_terminal_buffer_populated(self, synthetic_buffer):
        """The separate terminal_buffer deque must hold terminal entries."""
        assert len(synthetic_buffer.terminal_buffer) > 0

    def test_sample_16_does_not_crash(self, synthetic_buffer):
        """Sampling 16 transitions must not raise."""
        states, actions, rewards, next_states, dones, next_masks = synthetic_buffer.sample(16)
        assert states.shape == (16, 2, 6, 7)
        assert actions.shape == (16,)
        assert rewards.shape == (16,)


# ---------------------------------------------------------------------------
# 2. Training convergence checks
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    """Loss must decrease and remain finite across 500 steps."""

    def test_loss_history_has_500_values(self, trained_net_and_losses):
        _, loss_history = trained_net_and_losses
        assert len(loss_history) == 500

    def test_all_losses_are_finite(self, trained_net_and_losses):
        """No NaN or Inf at any step — catches gradient explosions."""
        _, loss_history = trained_net_and_losses
        for step, loss in enumerate(loss_history):
            assert np.isfinite(loss), f"Non-finite loss at step {step}: {loss}"

    def test_loss_decreases_early_to_late(self, trained_net_and_losses):
        """Mean loss over steps 451–500 must be lower than steps 1–50.

        This is the primary regression guard for 'network is learning'.
        Diagnostic baseline: early≈0.6, late≈0.005 at 500 steps.
        """
        _, loss_history = trained_net_and_losses
        early_mean = np.mean(loss_history[:50])
        late_mean  = np.mean(loss_history[-50:])
        assert late_mean < early_mean, (
            f"Loss did not decrease: early_mean={early_mean:.5f}, "
            f"late_mean={late_mean:.5f}"
        )

    def test_final_loss_below_0p1(self, trained_net_and_losses):
        """Mean loss over the final 50 steps must be < 0.10.

        At step 500 the diagnostic shows MSE ≈ 0.005, so 0.10 is a
        very generous threshold that should never trip unless something
        is fundamentally broken.
        """
        _, loss_history = trained_net_and_losses
        late_mean = np.mean(loss_history[-50:])
        assert late_mean < 0.10, (
            f"Final mean loss {late_mean:.5f} ≥ 0.10 — network not converging."
        )


# ---------------------------------------------------------------------------
# 3. Win Q-values: every single win transition must converge correctly
# ---------------------------------------------------------------------------

class TestWinQValues:
    """After 500 steps, every win-terminal Q(s,a) must be > 0 AND ≥ +0.5.

    Why 100%: all 22 win entries are terminal (done=True), so the Bellman
    target is exactly +1.0 with zero bootstrap noise.  A healthy network
    must fit all of them correctly on a buffer this small and clean.
    """

    def test_all_win_q_values_positive(self, synthetic_buffer, trained_net_and_losses):
        """100% of win-terminal Q(s,a_winning) must be > 0.0."""
        policy_net, _ = trained_net_and_losses
        wins = _collect_terminal_entries(synthetic_buffer, +1.0)
        assert len(wins) == 22
        q_vals = _eval_q_for_entries(policy_net, wins)
        failures = [(i, q) for i, q in enumerate(q_vals) if q <= 0.0]
        assert len(failures) == 0, (
            f"{len(failures)}/22 win Q-values are ≤ 0.  Failures: {failures}"
        )

    def test_all_win_q_values_at_least_half(self, synthetic_buffer, trained_net_and_losses):
        """100% of win-terminal Q(s,a_winning) must be ≥ +0.5.

        Diagnostic shows min = +0.926 at 500 steps, so +0.5 has large margin.
        """
        policy_net, _ = trained_net_and_losses
        wins = _collect_terminal_entries(synthetic_buffer, +1.0)
        assert len(wins) == 22
        q_vals = _eval_q_for_entries(policy_net, wins)
        failures = [(i, q) for i, q in enumerate(q_vals) if q < 0.5]
        assert len(failures) == 0, (
            f"{len(failures)}/22 win Q-values are < 0.5.  Failures: {failures}"
        )

    def test_mean_win_q_near_plus_one(self, synthetic_buffer, trained_net_and_losses):
        """Mean Q for win transitions must be ≥ +0.8 (diagnostic shows ≈ +0.997)."""
        policy_net, _ = trained_net_and_losses
        wins = _collect_terminal_entries(synthetic_buffer, +1.0)
        q_vals = _eval_q_for_entries(policy_net, wins)
        mean_q = float(np.mean(q_vals))
        assert mean_q >= 0.8, (
            f"Mean Q(win) = {mean_q:.4f}, expected ≥ 0.8"
        )


# ---------------------------------------------------------------------------
# 4. Loss Q-values: every single loss transition must converge correctly
# ---------------------------------------------------------------------------

class TestLossQValues:
    """After 500 steps, every loss-terminal Q(s,a) must be < 0 AND ≤ -0.5.

    Same rationale as win transitions: zero bootstrap noise, 100% is achievable
    and expected.
    """

    def test_all_loss_q_values_negative(self, synthetic_buffer, trained_net_and_losses):
        """100% of loss-terminal Q(s,a_losing) must be < 0.0."""
        policy_net, _ = trained_net_and_losses
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        assert len(losses) == 22
        q_vals = _eval_q_for_entries(policy_net, losses)
        failures = [(i, q) for i, q in enumerate(q_vals) if q >= 0.0]
        assert len(failures) == 0, (
            f"{len(failures)}/22 loss Q-values are ≥ 0.  Failures: {failures}"
        )

    def test_all_loss_q_values_at_most_minus_half(self, synthetic_buffer, trained_net_and_losses):
        """100% of loss-terminal Q(s,a_losing) must be ≤ -0.5.

        Diagnostic shows max = -0.949 at 500 steps, so -0.5 has large margin.
        """
        policy_net, _ = trained_net_and_losses
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        assert len(losses) == 22
        q_vals = _eval_q_for_entries(policy_net, losses)
        failures = [(i, q) for i, q in enumerate(q_vals) if q > -0.5]
        assert len(failures) == 0, (
            f"{len(failures)}/22 loss Q-values are > -0.5.  Failures: {failures}"
        )

    def test_mean_loss_q_near_minus_one(self, synthetic_buffer, trained_net_and_losses):
        """Mean Q for loss transitions must be ≤ -0.8 (diagnostic shows ≈ -0.990)."""
        policy_net, _ = trained_net_and_losses
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        q_vals = _eval_q_for_entries(policy_net, losses)
        mean_q = float(np.mean(q_vals))
        assert mean_q <= -0.8, (
            f"Mean Q(loss) = {mean_q:.4f}, expected ≤ -0.8"
        )


# ---------------------------------------------------------------------------
# 5. Win-vs-loss separation
# ---------------------------------------------------------------------------

class TestQValueSeparation:
    """Global ordering and spread of win vs loss Q-values."""

    def test_mean_win_greater_than_mean_loss(self, synthetic_buffer, trained_net_and_losses):
        """mean Q(win) > mean Q(loss) — fundamental sign ordering."""
        policy_net, _ = trained_net_and_losses
        wins   = _collect_terminal_entries(synthetic_buffer, +1.0)
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        win_qs  = _eval_q_for_entries(policy_net, wins)
        loss_qs = _eval_q_for_entries(policy_net, losses)
        assert np.mean(win_qs) > np.mean(loss_qs)

    def test_q_spread_at_least_1p6(self, synthetic_buffer, trained_net_and_losses):
        """Spread (mean_win - mean_loss) ≥ 1.6.

        With mean_win≈+0.997 and mean_loss≈-0.990, real spread ≈ 1.987.
        Threshold 1.6 leaves a comfortable margin for random seed variance
        while still catching a collapsed network.
        """
        policy_net, _ = trained_net_and_losses
        wins   = _collect_terminal_entries(synthetic_buffer, +1.0)
        losses = _collect_terminal_entries(synthetic_buffer, -1.0)
        win_qs  = _eval_q_for_entries(policy_net, wins)
        loss_qs = _eval_q_for_entries(policy_net, losses)
        spread = float(np.mean(win_qs)) - float(np.mean(loss_qs))
        assert spread >= 1.6, (
            f"Q spread = {spread:.4f}, expected ≥ 1.6. "
            f"mean_win={np.mean(win_qs):.4f}, mean_loss={np.mean(loss_qs):.4f}"
        )

    def test_no_nan_in_q_values(self, synthetic_buffer, trained_net_and_losses):
        """No NaN Q-values for any terminal transition."""
        policy_net, _ = trained_net_and_losses
        all_entries = (
            _collect_terminal_entries(synthetic_buffer, +1.0)
            + _collect_terminal_entries(synthetic_buffer, -1.0)
        )
        q_vals = _eval_q_for_entries(policy_net, all_entries)
        nans = [(i, q) for i, q in enumerate(q_vals) if np.isnan(q)]
        assert len(nans) == 0, f"NaN Q-values at indices: {nans}"
