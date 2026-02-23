"""
Phase 5 — Training Mechanics Smoke Test
=========================================

300 self-play episodes with the real DQN training loop.  This is a *mechanical*
test — it verifies plumbing correctness, not convergence to an optimal policy.

What we check (and why the bars are set where they are)
---------------------------------------------------------
1. No crash          — 300 episodes complete without exception.
2. Buffer fills      — len(buffer) > 2×BATCH_SIZE after 300 games.
3. Terminals exist   — terminal_buffer nonempty; games must end.
4. Loss is finite    — no NaN/Inf at any training step.
5. Loss doesn't explode — mean(last 50 losses) ≤ 3× mean(first 50 losses).
   300 episodes is too few to reliably *decrease* loss (state space is huge,
   epsilon=0.5 adds lots of noise), but it must not blow up.
6. Exploration happens — ≥50 unique board hashes across all 300 games.
7. Win rate vs random ≥ 30% — greedy DQN vs pure-random opponent, 30 eval
   games.  A completely untrained (random weights) net wins ~50% because it
   effectively plays randomly too; a minimally-trained net should be able to
   maintain ≥30% after 300 episodes.

Self-play loop (extracted from train_dqn_20251221.py)
------------------------------------------------------
- add_symmetric() doubles each transition (original + horizontal mirror).
- update_penalty(-3, -1.0, True) + update_penalty(-4, -1.0, True) mark the
  opponent's second-to-last move as a loss.
- TRAIN_N_TIMES_PER_GAME=4 gradient steps per episode once buffer is ready.
- NegaMax Bellman: target = r - gamma * max(Q_target(s')) * (1 - done)
- Target net synced every TARGET_UPDATE_FREQ=100 episodes.
"""

import copy
import sys
import hashlib
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, ".")

from src.networks.connect4_net import Connect4Net
from src.environment.connect4 import ConnectFourEnvironment
from src.environment.config import Config
from src.utils.dqn_replay_buffer import DQNReplayBuffer


# ---------------------------------------------------------------------------
# Helpers (self-contained, no dependency on the monolithic training script)
# ---------------------------------------------------------------------------

def _get_action_mask(legal_moves: List[int]) -> np.ndarray:
    mask = np.zeros(7, dtype=np.float32)
    mask[legal_moves] = 1.0
    return mask


def _select_action(policy_net: Connect4Net, state: np.ndarray,
                   legal_moves: List[int], eps: float) -> int:
    """Epsilon-greedy action selection with illegal-move masking."""
    if np.random.random() < eps:
        return int(np.random.choice(legal_moves))
    policy_net.eval()
    with torch.no_grad():
        q = policy_net(state).squeeze(0)
        illegal = list(set(range(7)) - set(legal_moves))
        q[illegal] = -1e9
        return int(torch.argmax(q).item())


def _state_hash(state: np.ndarray) -> str:
    return hashlib.md5(state.tobytes()).hexdigest()


def _play_one_game(env: ConnectFourEnvironment,
                   policy_net: Connect4Net,
                   replay_buffer: DQNReplayBuffer,
                   eps: float):
    """Play one self-play game; push transitions into replay_buffer."""
    env.reset()
    done = False
    moves = 0
    reward = 0.0

    while not done and moves < 42:
        state = env.get_state()
        legal_moves = env.get_legal_moves()
        action = _select_action(policy_net, state, legal_moves, eps)

        next_state, reward, done = env.play_move(action)

        next_mask = (
            _get_action_mask(env.get_legal_moves())
            if not done
            else np.zeros(7, dtype=np.float32)
        )
        replay_buffer.add_symmetric(state, action, reward, next_state, done, next_mask)
        moves += 1

    # Second-to-last move penalty (mirrors train_dqn_20251221.py)
    if reward == 1.0:
        replay_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)
        replay_buffer.update_penalty(index=-4, new_reward=-1.0, is_done=True)

    return reward


def _training_step(policy_net: Connect4Net,
                   target_net: Connect4Net,
                   replay_buffer: DQNReplayBuffer,
                   optimizer: torch.optim.Optimizer,
                   config: Config,
                   device: str = "cpu") -> float:
    """One gradient step. Returns loss value."""
    states, actions, rewards, next_states, dones, next_masks = replay_buffer.sample(
        config.BATCH_SIZE, terminal_ratio=config.TERMINAL_RATE
    )

    s  = torch.tensor(states,      dtype=torch.float32, device=device)
    a  = torch.tensor(actions,     dtype=torch.long,    device=device)
    r  = torch.tensor(rewards,     dtype=torch.float32, device=device)
    ns = torch.tensor(next_states, dtype=torch.float32, device=device)
    d  = torch.tensor(dones,       dtype=torch.float32, device=device)
    m  = torch.tensor(next_masks,  dtype=torch.float32, device=device)

    with torch.no_grad():
        nq = target_net(ns)
        nq.masked_fill_(m == 0, -1e9)
        target_q = r - config.GAMMA * nq.max(1)[0] * (1 - d)

    policy_net.train()
    optimizer.zero_grad()
    qv = policy_net(s)
    pq = qv.gather(1, a.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(pq, target_q)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


def _evaluate_vs_random(policy_net: Connect4Net,
                        config: Config,
                        num_games: int = 30) -> float:
    """Win rate of greedy DQN (player 1) vs random player (player 2)."""
    env = ConnectFourEnvironment(config)
    wins = 0
    policy_net.eval()

    for _ in range(num_games):
        env.reset()
        done = False
        moves = 0
        while not done and moves < 42:
            state = env.get_state()
            legal_moves = env.get_legal_moves()
            current_player = env.get_current_player()

            if current_player == 1:
                action = _select_action(policy_net, state, legal_moves, eps=0.0)
            else:
                action = int(np.random.choice(legal_moves))

            _, reward, done = env.play_move(action)
            moves += 1

        # reward is from last player's perspective; if player 1 made the last
        # move and reward==1.0, player 1 won.  Track wins for player 1.
        winner = env.check_winner()
        if winner == 1:
            wins += 1

    return wins / num_games


# ---------------------------------------------------------------------------
# Module-scoped fixture: run 300 training episodes once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def smoke_run():
    """
    Run 300 self-play episodes.

    Returns dict with:
        policy_net, config, replay_buffer,
        loss_history (list[float]),
        unique_state_hashes (set),
        win_rate_vs_random (float)
    """
    config = Config()
    device = "cpu"

    policy_net = Connect4Net(device=device, dropout_rate=config.DROPOUT_RATE)
    target_net = copy.deepcopy(policy_net)
    target_net.eval()

    replay_buffer = DQNReplayBuffer(capacity=10_000)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    env = ConnectFourEnvironment(config)

    NUM_EPISODES = 300
    eps = config.EPS_START
    loss_history: list[float] = []
    unique_hashes: set = set()

    for episode in range(1, NUM_EPISODES + 1):
        # Collect one game
        _play_one_game(env, policy_net, replay_buffer, eps)

        # Track explored states via env's last game board positions
        for entry in list(replay_buffer.buffer)[-10:]:   # last 10 added entries
            unique_hashes.add(_state_hash(np.array(entry.state)))

        # Train TRAIN_N_TIMES_PER_GAME times if buffer is ready
        if replay_buffer.is_ready(config.BATCH_SIZE):
            for _ in range(config.TRAIN_N_TIMES_PER_GAME):
                loss_val = _training_step(
                    policy_net, target_net, replay_buffer, optimizer, config, device
                )
                loss_history.append(loss_val)

        # Epsilon decay
        eps = max(config.EPS_END, eps * config.EPS_DECAY)

        # Target net sync
        if episode % config.TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Progress print every 50 episodes so the test run doesn't look frozen
        if episode % 50 == 0:
            recent_loss = np.mean(loss_history[-20:]) if len(loss_history) >= 20 else float("nan")
            print(
                f"  [smoke] ep={episode:>3}/{NUM_EPISODES}  "
                f"buf={len(replay_buffer):>5}  "
                f"terminals={len(replay_buffer.terminal_buffer):>4}  "
                f"eps={eps:.3f}  "
                f"loss(last20)={recent_loss:.3f}"
            )

    win_rate = _evaluate_vs_random(policy_net, config, num_games=30)

    return {
        "policy_net": policy_net,
        "config": config,
        "replay_buffer": replay_buffer,
        "loss_history": loss_history,
        "unique_hashes": unique_hashes,
        "win_rate_vs_random": win_rate,
    }


# ---------------------------------------------------------------------------
# 1. Mechanical completeness
# ---------------------------------------------------------------------------

class TestMechanicalCompleteness:
    """300 episodes must complete without exception and produce data."""

    def test_smoke_run_completes(self, smoke_run):
        """The fixture itself completing is the test — no exception thrown."""
        assert smoke_run is not None

    def test_loss_history_nonempty(self, smoke_run):
        """Training steps must have fired (buffer must have filled)."""
        assert len(smoke_run["loss_history"]) > 0, (
            "No training steps were taken — buffer never reached BATCH_SIZE"
        )

    def test_training_steps_fired_multiple_times(self, smoke_run):
        """Expect at least 100 training steps across 300 episodes."""
        n = len(smoke_run["loss_history"])
        assert n >= 100, (
            f"Only {n} training steps — expected ≥100 across 300 episodes"
        )


# ---------------------------------------------------------------------------
# 2. Replay buffer health
# ---------------------------------------------------------------------------

class TestReplayBufferHealth:
    """Buffer must fill with diverse, terminal-rich data."""

    def test_buffer_exceeds_twice_batch_size(self, smoke_run):
        """Buffer must hold at least 2×BATCH_SIZE entries."""
        buf = smoke_run["replay_buffer"]
        config = smoke_run["config"]
        assert len(buf) >= 2 * config.BATCH_SIZE, (
            f"Buffer size {len(buf)} < 2×BATCH_SIZE={2*config.BATCH_SIZE}"
        )

    def test_terminal_buffer_nonempty(self, smoke_run):
        """Games must end — terminal_buffer must be populated."""
        buf = smoke_run["replay_buffer"]
        assert len(buf.terminal_buffer) > 0, (
            "terminal_buffer is empty — games may not be ending properly"
        )

    def test_terminal_buffer_has_at_least_10_entries(self, smoke_run):
        """Expect many terminal entries across 300 games."""
        buf = smoke_run["replay_buffer"]
        n = len(buf.terminal_buffer)
        assert n >= 10, (
            f"Only {n} terminal entries after 300 games — expected ≥10"
        )

    def test_buffer_contains_both_rewards(self, smoke_run):
        """Buffer must have both +1.0 and -1.0 terminal rewards."""
        buf = smoke_run["replay_buffer"]
        rewards = [float(e.reward) for e in buf.terminal_buffer]
        has_win  = any(r > 0.5 for r in rewards)
        has_loss = any(r < -0.5 for r in rewards)
        assert has_win,  "No +1.0 win entries in terminal_buffer"
        assert has_loss, "No -1.0 loss entries in terminal_buffer (update_penalty not firing?)"


# ---------------------------------------------------------------------------
# 3. Loss stability
# ---------------------------------------------------------------------------

class TestLossStability:
    """Loss must remain finite and must not explode."""

    def test_all_losses_finite(self, smoke_run):
        """No NaN or Inf at any training step."""
        for step, loss in enumerate(smoke_run["loss_history"]):
            assert np.isfinite(loss), f"Non-finite loss at step {step}: {loss}"

    def test_loss_does_not_explode(self, smoke_run):
        """Mean loss over the second half ≤ 3× mean over the first half.

        300 episodes is far too few to reliably *decrease* loss given the
        large state space and epsilon=0.5 noise, but it must not blow up.
        Threshold 3× is intentionally generous.
        """
        lh = smoke_run["loss_history"]
        if len(lh) < 20:
            pytest.skip("Too few training steps to compare halves")
        mid = len(lh) // 2
        first_mean = np.mean(lh[:mid])
        second_mean = np.mean(lh[mid:])
        assert second_mean <= 3.0 * first_mean + 1e-6, (
            f"Loss exploded: first_half_mean={first_mean:.4f}, "
            f"second_half_mean={second_mean:.4f} (ratio={second_mean/first_mean:.1f}×)"
        )

    def test_second_half_loss_below_50(self, smoke_run):
        """Mean loss in the second half of training must be < 50.

        The first training steps see large MSE losses because the random-init
        target network produces Q-values of ±10–20, making Bellman targets
        like ±15 and MSE from zero ≈ 225.  After the first target-net sync
        these stabilise.  Asserting mean(second half) < 50 catches genuine
        divergence while tolerating the normal early-training spike.
        """
        lh = smoke_run["loss_history"]
        if len(lh) < 20:
            pytest.skip("Too few training steps")
        mid = len(lh) // 2
        second_half_mean = np.mean(lh[mid:])
        assert second_half_mean < 50.0, (
            f"Second-half mean loss {second_half_mean:.2f} ≥ 50 — "
            f"network may be diverging after initial stabilisation"
        )


# ---------------------------------------------------------------------------
# 4. Exploration
# ---------------------------------------------------------------------------

class TestExploration:
    """Agent must explore a diverse set of board states."""

    def test_unique_states_at_least_50(self, smoke_run):
        """At least 50 unique board hashes across 300 games.

        With epsilon=0.5 and a 6×7 board, this is a very low bar.
        Failure means the agent is stuck in a degenerate loop.
        """
        n = len(smoke_run["unique_hashes"])
        assert n >= 50, (
            f"Only {n} unique states seen — agent may be stuck in a loop"
        )


# ---------------------------------------------------------------------------
# 5. Win rate vs random
# ---------------------------------------------------------------------------

class TestWinRateVsRandom:
    """Greedy DQN must win at least 30% of games against a random opponent.

    Rationale:
    - A random-weight DQN (purely random-looking play) wins ~50% vs random
      due to first-player advantage; however once epsilon-greedy stops being
      fully random, a poorly-trained network can actually perform worse.
    - 30% is a floor that should be met even with minimal training.
    - Target for Phase 6 is >90%.
    """

    def test_win_rate_at_least_30_percent(self, smoke_run):
        wr = smoke_run["win_rate_vs_random"]
        assert wr >= 0.30, (
            f"Win rate vs random = {wr:.0%} — expected ≥ 30% after 300 episodes"
        )

    def test_win_rate_value_in_range(self, smoke_run):
        """Win rate must be a valid probability."""
        wr = smoke_run["win_rate_vs_random"]
        assert 0.0 <= wr <= 1.0, f"Win rate out of range: {wr}"
