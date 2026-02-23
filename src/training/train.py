"""
src/training/train.py
=====================
Full DQN training script for Connect 4.

Usage
-----
    python -m src.training.train                        # default 10 000 episodes
    python -m src.training.train --episodes 50000
    python -m src.training.train --episodes 20000 --lr 3e-4

What it does
------------
1. Prints a hyperparameter table so you know exactly what is running.
2. Self-play loop with epsilon-greedy exploration + symmetric experience storage.
3. Every 100 episodes  → one compact progress line (loss, buffer size, epsilon).
4. Every 500 episodes  → full evaluation vs random agent (50 games) + unique states.
5. Saves the *best* model (highest win rate seen) to models/ as it trains.
6. Saves a final snapshot at the end.
7. Automatic early-stop if:
   - Mean loss (last 20 updates) > 500 for 3 consecutive 100-ep checks  →  "LOSS EXPLODED"
   - Win rate = 0 % for 3 consecutive evaluations after episode 2 000   →  "WIN RATE COLLAPSED"

Device
------
Device is auto-selected by src/utils/device.py:
  MPS (Apple Silicon) if all safety checks pass, otherwise CPU.
"""

import argparse
import copy
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.environment.config import Config
from src.environment.connect4 import ConnectFourEnvironment
from src.networks.connect4_net import Connect4Net
from src.utils.dqn_replay_buffer import DQNReplayBuffer


# ---------------------------------------------------------------------------
# Hyper-parameters (overridable from CLI or by editing the dict below)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    total_episodes        = 10_000,
    lr                    = 1e-4,
    eps_start             = 1.0,
    eps_end               = 0.05,
    eps_decay             = 0.9999,
    batch_size            = 128,
    gamma                 = 0.99,
    terminal_ratio        = 0.25,   # fraction of each batch that must be terminal
    target_sync_every     = 500,    # gradient steps between target-net sync
    train_n_times         = 4,      # gradient updates per episode
    buffer_capacity       = 50_000,
    min_buffer            = 1_000,  # don't start training until buffer reaches this
    eval_every            = 500,    # episodes between evaluations vs random
    eval_games            = 50,     # games per evaluation
    print_every           = 100,    # episodes between compact progress lines
    dropout_rate          = 0.0,
    opponent_greedy_frac  = 0.5,    # fraction of games where opponent plays greedy best model
)

# Early-stop thresholds
LOSS_EXPLODE_THRESHOLD   = 500.0   # mean loss > this for N consecutive checks
LOSS_EXPLODE_STRIKES     = 3       # consecutive checks before stopping
WIN_COLLAPSE_THRESHOLD   = 0.0     # win rate at or below this
WIN_COLLAPSE_STRIKES     = 3       # consecutive evaluations before stopping
WIN_COLLAPSE_AFTER_EP    = 2_000   # only check for collapse after this many eps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_mask(legal_moves: List[int], cols: int = 7) -> np.ndarray:
    """Binary mask: 1 = legal, 0 = illegal."""
    mask = np.zeros(cols, dtype=np.int16)
    for m in legal_moves:
        mask[m] = 1
    return mask


def _select_action(
    net: Connect4Net,
    state: np.ndarray,
    legal_moves: List[int],
    eps: float,
    device: str,
) -> int:
    """Epsilon-greedy with illegal-move masking."""
    if random.random() < eps or not legal_moves:
        return random.choice(legal_moves)

    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = net(s).squeeze(0).cpu().numpy()          # (7,)

    # Mask illegal columns to -inf before argmax
    masked = np.full(len(q), -1e9, dtype=np.float32)
    for m in legal_moves:
        masked[m] = q[m]
    return int(np.argmax(masked))


def _play_one_game(
    env: ConnectFourEnvironment,
    policy_net: Connect4Net,
    buf: DQNReplayBuffer,
    eps: float,
    device: str,
    opponent_net: Optional[Connect4Net] = None,
) -> int:
    """
    One self-play episode.  Returns the number of moves played.
    Stores transitions with add_symmetric (data augmentation via horizontal flip).
    Also applies the NegaMax penalty: the move *before* a loss gets reward -1.

    Args:
        opponent_net: If provided, the second player (P2) plays greedy using this
                      frozen network (eps=0). If None, P2 uses policy_net with
                      the same eps as P1 (original self-play behaviour).
    """
    state = env.reset()
    done = False
    move_count = 0
    # Track last transition per player for retroactive penalty
    last_transition: dict = {}   # player -> index_in_buf

    # P1 = trainee (always policy_net + eps)
    # P2 = opponent (greedy best_net if provided, else policy_net + eps)
    trainee_player  = env.get_current_player()   # P1 goes first

    while not done:
        player = env.get_current_player()
        legal  = env.get_legal_moves()
        if not legal:
            break

        # Choose network and epsilon for this player
        if player == trainee_player or opponent_net is None:
            action = _select_action(policy_net, state, legal, eps, device)
        else:
            # Greedy opponent — eps=0, uses frozen best-model copy
            action = _select_action(opponent_net, state, legal, 0.0, device)

        next_state, reward, done = env.play_move(action)
        next_legal = env.get_legal_moves()
        next_mask  = _action_mask(next_legal)

        # Retroactive penalty: if the current player just won, the *previous*
        # move by the opponent (which failed to prevent this) was a loss.
        if done and reward == 1.0:
            opponent = -player
            if opponent in last_transition:
                buf.update_penalty(last_transition[opponent], -1.0, True)

        buf.add_symmetric(state, action, reward, next_state, done, next_mask)
        # add_symmetric adds 2 entries; the original is at index -2
        last_transition[player] = -2

        state = next_state
        move_count += 1

    return move_count


def _training_step(
    policy_net: Connect4Net,
    target_net: Connect4Net,
    buf: DQNReplayBuffer,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    device: str,
) -> float:
    """One mini-batch gradient update. Returns scalar loss."""
    if not buf.is_ready(cfg["batch_size"]):
        return 0.0

    states, actions, rewards, next_states, dones, next_masks = buf.sample(
        cfg["batch_size"],
        terminal_ratio=cfg["terminal_ratio"],
    )

    states_t      = torch.from_numpy(states).float().to(device)
    actions_t     = torch.from_numpy(actions).long().to(device)
    rewards_t     = torch.from_numpy(rewards).float().to(device)
    next_states_t = torch.from_numpy(next_states).float().to(device)
    dones_t       = torch.from_numpy(dones).float().to(device)
    masks_t       = torch.from_numpy(next_masks.astype(np.float32)).to(device)

    # Current Q estimates
    policy_net.train()
    q_all  = policy_net(states_t)                            # (B, 7)
    q_pred = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)   # (B,)

    # Target Q  (NegaMax-style: opponent maximises, so we negate)
    with torch.no_grad():
        q_next_all = target_net(next_states_t)               # (B, 7)
        # Mask illegal next actions to -inf before max
        q_next_all = q_next_all + (masks_t - 1) * 1e9
        q_next_max = q_next_all.max(dim=1).values            # (B,)
        target_q   = rewards_t - cfg["gamma"] * q_next_max * (1.0 - dones_t)

    loss = F.mse_loss(q_pred, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return float(loss.item())


def _evaluate_vs_random(
    policy_net: Connect4Net,
    n_games: int,
    device: str,
) -> float:
    """
    Play n_games where policy_net is P1 (greedy, eps=0) vs a random P2.
    Returns win fraction for the policy net.
    """
    cfg_eval = Config()
    env = ConnectFourEnvironment(cfg_eval)
    wins = 0

    policy_net.eval()
    for _ in range(n_games):
        state = env.reset()
        done = False

        while not done:
            player = env.get_current_player()
            legal  = env.get_legal_moves()
            if not legal:
                break

            if player == cfg_eval.PLAYER_1:
                # Greedy policy
                action = _select_action(policy_net, state, legal, eps=0.0, device=device)
            else:
                # Random opponent
                action = random.choice(legal)

            state, reward, done = env.play_move(action)

            if done and reward == 1.0 and player == cfg_eval.PLAYER_1:
                wins += 1

    policy_net.train()
    return wins / n_games


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _print_banner(cfg: dict, device: str) -> None:
    """Print a clean hyperparameter table."""
    sep = "=" * 60
    print(sep)
    print("  Connect 4 DQN Training  [greedy-opponent mode]")
    print(sep)
    print(f"  Device             : {device}")
    print(f"  Total episodes     : {cfg['total_episodes']:,}")
    print(f"  Learning rate      : {cfg['lr']}")
    print(f"  Epsilon            : {cfg['eps_start']} → {cfg['eps_end']}  (decay {cfg['eps_decay']} per step)")
    print(f"  Batch size         : {cfg['batch_size']}")
    print(f"  Gamma (discount)   : {cfg['gamma']}")
    print(f"  Terminal ratio     : {cfg['terminal_ratio']:.0%}  (of each batch are terminal transitions)")
    print(f"  Target sync every  : {cfg['target_sync_every']} gradient steps")
    print(f"  Train N times/game : {cfg['train_n_times']}")
    print(f"  Buffer capacity    : {cfg['buffer_capacity']:,}")
    print(f"  Min buffer to train: {cfg['min_buffer']:,}")
    print(f"  Eval every         : {cfg['eval_every']} episodes  ({cfg['eval_games']} games each)")
    print(f"  Dropout rate       : {cfg['dropout_rate']}")
    print(f"  Greedy opponent    : {cfg['opponent_greedy_frac']:.0%} of games use frozen best-model as P2")
    print(sep)
    print(f"  Early stop: loss > {LOSS_EXPLODE_THRESHOLD} for {LOSS_EXPLODE_STRIKES} checks  OR")
    print(f"              win rate = 0% for {WIN_COLLAPSE_STRIKES} evals after ep {WIN_COLLAPSE_AFTER_EP:,}")
    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: Optional[dict] = None) -> dict:
    """
    Run the full DQN training loop.

    Args:
        cfg: Override dictionary. Keys matching DEFAULTS will replace defaults.
             Pass None to use all defaults.

    Returns:
        dict with keys: policy_net, best_win_rate, final_model_path, loss_history,
        win_rate_history, total_time_s
    """
    # ---- Merge config -------------------------------------------------------
    run_cfg = {**DEFAULTS}
    if cfg:
        run_cfg.update(cfg)

    device = Config.DEVICE

    _print_banner(run_cfg, device)

    # ---- Networks -----------------------------------------------------------
    policy_net = Connect4Net(device=device, dropout_rate=run_cfg["dropout_rate"])
    target_net = Connect4Net(device=device, dropout_rate=0.0)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Frozen "best model so far" used as greedy opponent.
    # Starts as a copy of the randomly-initialised policy_net;
    # updated each time the policy achieves a new best win rate.
    best_net = Connect4Net(device=device, dropout_rate=0.0)
    best_net.load_state_dict(policy_net.state_dict())
    best_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=run_cfg["lr"])

    # ---- Buffer & environment -----------------------------------------------
    buf = DQNReplayBuffer(capacity=run_cfg["buffer_capacity"])
    game_env = ConnectFourEnvironment(Config())

    # ---- Tracking -----------------------------------------------------------
    eps              = run_cfg["eps_start"]
    total_grad_steps = 0
    loss_window      = deque(maxlen=20)   # rolling window for recent losses
    all_losses: List[float] = []
    win_rate_history: List[tuple] = []    # (episode, win_rate)
    unique_states: set = set()
    best_win_rate    = 0.0
    best_model_path: Optional[Path] = None

    # Early-stop counters
    loss_strike      = 0
    win_collapse_strike = 0

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"  {'EP':>6}  {'BUF':>6}  {'TERM':>6}  {'EPS':>6}  {'LOSS(20)':>10}  {'GRAD':>6}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*6}")

    t_start = time.time()

    for episode in range(1, run_cfg["total_episodes"] + 1):

        # ---- Play one game --------------------------------------------------
        # With probability opponent_greedy_frac, P2 is the frozen best model
        # (greedy, eps=0). Otherwise pure self-play (original behaviour).
        use_greedy = random.random() < run_cfg["opponent_greedy_frac"]
        _play_one_game(
            game_env, policy_net, buf, eps, device,
            opponent_net=best_net if use_greedy else None,
        )

        # Track unique states visited (hash board for uniqueness check)
        state_snapshot = game_env.get_state()
        unique_states.add(state_snapshot.tobytes())

        # ---- Decay epsilon --------------------------------------------------
        eps = max(run_cfg["eps_end"], eps * run_cfg["eps_decay"])

        # ---- Training steps -------------------------------------------------
        if buf.is_ready(run_cfg["min_buffer"]):
            for _ in range(run_cfg["train_n_times"]):
                loss_val = _training_step(
                    policy_net, target_net, buf, optimizer, run_cfg, device
                )
                total_grad_steps += 1
                if loss_val > 0:
                    loss_window.append(loss_val)
                    all_losses.append(loss_val)

                # Sync target net
                if total_grad_steps % run_cfg["target_sync_every"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        # ---- Compact progress line every print_every episodes ---------------
        if episode % run_cfg["print_every"] == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            n_terminals = len(buf.terminal_buffer)
            print(
                f"  ep={episode:>6,d}  "
                f"buf={len(buf):>6,d}  "
                f"term={n_terminals:>5,d}  "
                f"eps={eps:.4f}  "
                f"loss(20)={mean_loss:>9.3f}  "
                f"grad={total_grad_steps:>6,d}"
            )

            # ---- Early stop: loss explosion ---------------------------------
            if loss_window and mean_loss > LOSS_EXPLODE_THRESHOLD:
                loss_strike += 1
                print(f"  ⚠️  Loss spike #{loss_strike}/{LOSS_EXPLODE_STRIKES}  (mean={mean_loss:.1f})")
                if loss_strike >= LOSS_EXPLODE_STRIKES:
                    print(f"\n  ❌ LOSS EXPLODED — stopping training at episode {episode}.")
                    break
            else:
                loss_strike = 0

        # ---- Full evaluation every eval_every episodes ----------------------
        if episode % run_cfg["eval_every"] == 0:
            t_eval = time.time()
            win_rate = _evaluate_vs_random(policy_net, run_cfg["eval_games"], device)
            t_eval   = time.time() - t_eval
            n_unique = len(unique_states)

            win_rate_history.append((episode, win_rate))

            marker = ""
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                # Save best model
                best_model_path = models_dir / f"dqn_connect4_best_{timestamp}.pth"
                torch.save(policy_net.state_dict(), best_model_path)
                # Update the frozen opponent to the new best weights
                best_net.load_state_dict(policy_net.state_dict())
                best_net.eval()
                marker = "  ← best (opponent updated)"

            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            elapsed   = time.time() - t_start

            print()
            print(
                f"  ── EVAL ep={episode:>6,d} ──  "
                f"win%={win_rate:.1%}  "
                f"unique_states={n_unique:,}  "
                f"loss={mean_loss:.3f}  "
                f"eval_t={t_eval:.1f}s  "
                f"total={elapsed/60:.1f}min"
                + marker
            )
            print()

            # ---- Early stop: win rate collapse ------------------------------
            if episode > WIN_COLLAPSE_AFTER_EP and win_rate <= WIN_COLLAPSE_THRESHOLD:
                win_collapse_strike += 1
                print(f"  ⚠️  Win collapse #{win_collapse_strike}/{WIN_COLLAPSE_STRIKES}")
                if win_collapse_strike >= WIN_COLLAPSE_STRIKES:
                    print(
                        f"\n  ❌ WIN RATE COLLAPSED (0% for {WIN_COLLAPSE_STRIKES} evals) "
                        f"— stopping at episode {episode}."
                    )
                    break
            else:
                win_collapse_strike = 0

    # ---- Final model save ---------------------------------------------------
    final_path = models_dir / f"dqn_connect4_final_{timestamp}.pth"
    torch.save(
        {
            "state_dict"   : policy_net.state_dict(),
            "episode"      : episode,
            "best_win_rate": best_win_rate,
            "eps"          : eps,
            "total_grad_steps": total_grad_steps,
            "hyperparams"  : run_cfg,
            "device"       : device,
            "timestamp"    : timestamp,
        },
        final_path,
    )

    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Training complete in {total_time/60:.1f} min  ({total_time:.0f}s)")
    print(f"  Episodes run       : {episode:,} / {run_cfg['total_episodes']:,}")
    print(f"  Gradient steps     : {total_grad_steps:,}")
    print(f"  Best win rate      : {best_win_rate:.1%}")
    print(f"  Final epsilon      : {eps:.4f}")
    print(f"  Unique states seen : {len(unique_states):,}")
    print(f"  Final model saved  : {final_path}")
    if best_model_path:
        print(f"  Best model saved   : {best_model_path}")
    print("=" * 60)

    return dict(
        policy_net        = policy_net,
        best_win_rate     = best_win_rate,
        final_model_path  = str(final_path),
        best_model_path   = str(best_model_path) if best_model_path else None,
        loss_history      = all_losses,
        win_rate_history  = win_rate_history,
        total_time_s      = total_time,
        total_episodes    = episode,
        total_grad_steps  = total_grad_steps,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN agent for Connect 4")
    p.add_argument("--episodes",     type=int,   default=DEFAULTS["total_episodes"])
    p.add_argument("--lr",           type=float, default=DEFAULTS["lr"])
    p.add_argument("--eps-start",    type=float, default=DEFAULTS["eps_start"])
    p.add_argument("--eps-end",      type=float, default=DEFAULTS["eps_end"])
    p.add_argument("--eps-decay",    type=float, default=DEFAULTS["eps_decay"])
    p.add_argument("--batch-size",   type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--gamma",        type=float, default=DEFAULTS["gamma"])
    p.add_argument("--train-n",      type=int,   default=DEFAULTS["train_n_times"],
                   help="Gradient updates per episode")
    p.add_argument("--target-sync",  type=int,   default=DEFAULTS["target_sync_every"],
                   help="Gradient steps between target-net sync")
    p.add_argument("--eval-every",   type=int,   default=DEFAULTS["eval_every"])
    p.add_argument("--eval-games",   type=int,   default=DEFAULTS["eval_games"])
    p.add_argument("--buffer",       type=int,   default=DEFAULTS["buffer_capacity"])
    p.add_argument("--print-every",  type=int,   default=DEFAULTS["print_every"],
                   help="Episodes between compact progress lines")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg_override = dict(
        total_episodes   = args.episodes,
        lr               = args.lr,
        eps_start        = args.eps_start,
        eps_end          = args.eps_end,
        eps_decay        = args.eps_decay,
        batch_size       = args.batch_size,
        gamma            = args.gamma,
        train_n_times    = args.train_n,
        target_sync_every= args.target_sync,
        eval_every       = args.eval_every,
        eval_games       = args.eval_games,
        buffer_capacity  = args.buffer,
        print_every      = args.print_every,
    )
    train(cfg_override)
