"""
src/training/train.py
=====================
Full DQN training script for Connect 4.

Usage
-----
    python -m src.training.train                        # default 10 000 episodes
    python -m src.training.train --episodes 100000
    python -m src.training.train --episodes 50000 --lr 3e-4

Architecture overview
---------------------
The environment uses a *canonical* state representation (see connect4.py):
  - Channel 0: "MY pieces"  (always the current mover)
  - Channel 1: "OPPONENT's pieces"
  - After every move, the board is automatically flipped so the next mover
    always sees it from their own perspective.

Because the state is canonical, **one network can play both sides identically**.
The training loop therefore uses policy_net for ALL moves in every game:
  - No "Player 1 net" vs "Player 2 net" distinction.
  - No opponent pool playing moves during training.
  - Exploration is handled entirely by epsilon-greedy (eps decays 1.0 → 0.05).

Three networks, three distinct roles
-------------------------------------
policy_net  (live weights)
    Selects moves during training episodes (eps-greedy).
    Updated every gradient step.
    The "challenger" — it is always improving.

target_net  (lagged copy of policy_net)
    Used ONLY in the Bellman update to compute bootstrap targets Q(s', a').
    Synced from policy_net every target_sync_every gradient steps.
    Prevents the gradient update from chasing a moving target.

champion_net  (frozen copy, updated only on promotion)
    The *quality gate* — not used during training games.
    Every champion_eval_every episodes, policy_net (challenger) plays
    champion_net in champion_eval_games head-to-head games (both sides),
    using Boltzmann sampling with a low temperature (champion_eval_tau ≈ 0.2)
    so that each game explores different lines and the win-rate estimate is
    statistically meaningful.
    If the challenger wins > champion_win_thresh of those games it is promoted:
      champion_net ← policy_net, saved to disk, event logged.

Device
------
Auto-selected by src/utils/device.py:
  MPS (Apple Silicon) if all safety checks pass, otherwise CPU.
"""

import argparse
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
# Hyper-parameters  (all overridable from CLI or by passing a dict to train())
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
    eval_games            = 50,     # games per evaluation vs random
    print_every           = 100,    # episodes between compact progress lines
    dropout_rate          = 0.1,    # regularisation (0.0 = off, 0.1–0.2 typical)

    # ---- Champion-Challenger -------------------------------------------------
    champion_eval_every   = 2_000,  # episodes between challenger vs champion bouts
    champion_eval_games   = 100,    # games per side per bout (total = 2 × this)
    champion_win_thresh   = 0.55,   # challenger must win > this fraction to promote
    champion_eval_tau     = 0.2,    # Boltzmann temperature for bout games
                                    # τ → 0: near-argmax  τ → ∞: random
                                    # 0.2 keeps moves near-greedy but ensures
                                    # each game explores different lines
)

# Early-stop thresholds
LOSS_EXPLODE_THRESHOLD   = 500.0
LOSS_EXPLODE_STRIKES     = 3
WIN_COLLAPSE_THRESHOLD   = 0.0
WIN_COLLAPSE_STRIKES     = 3
WIN_COLLAPSE_AFTER_EP    = 2_000


# ---------------------------------------------------------------------------
# Action-selection helpers
# ---------------------------------------------------------------------------

def _action_mask(legal_moves: List[int], cols: int = 7) -> np.ndarray:
    """Binary mask: 1 = legal, 0 = illegal."""
    mask = np.zeros(cols, dtype=np.int16)
    for m in legal_moves:
        mask[m] = 1
    return mask


def _select_action_eps_greedy(
    net: Connect4Net,
    state: np.ndarray,
    legal_moves: List[int],
    eps: float,
    device: str,
) -> int:
    """
    Epsilon-greedy action selection with illegal-move masking.
    Used for policy_net during training and for greedy evaluation.
    """
    if random.random() < eps or not legal_moves:
        return random.choice(legal_moves)

    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = net(s).squeeze(0).cpu().numpy()   # (7,)

    masked = np.full(len(q), -1e9, dtype=np.float32)
    for m in legal_moves:
        masked[m] = q[m]
    return int(np.argmax(masked))


def _select_action_boltzmann(
    net: Connect4Net,
    state: np.ndarray,
    legal_moves: List[int],
    tau: float,
    device: str,
) -> int:
    """
    Boltzmann (softmax) action selection.

    Converts Q-values to a probability distribution via softmax(Q / τ),
    then samples.  Used for champion-challenger evaluation games so that:
      - Each of the 100 games explores different board lines
        (pure argmax would collapse to 1 deterministic game per side)
      - Moves are still strongly biased toward high-Q columns (τ is small)

    τ = 0.2 → near-greedy, small random variation each game
    τ = 1.0 → proportional to Q-values
    τ → ∞   → uniform random
    """
    if tau <= 0 or not legal_moves:
        return _select_action_eps_greedy(net, state, legal_moves, eps=0.0, device=device)

    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = net(s).squeeze(0).cpu()   # (7,)

    legal_t = torch.tensor(legal_moves, dtype=torch.long)
    q_legal = q[legal_t]
    probs   = torch.softmax(q_legal / tau, dim=0).numpy()
    idx     = int(np.random.choice(len(legal_moves), p=probs))
    return legal_moves[idx]


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def _play_one_game(
    env: ConnectFourEnvironment,
    policy_net: Connect4Net,
    buf: DQNReplayBuffer,
    eps: float,
    device: str,
) -> int:
    """
    One training episode.  Returns the number of moves played.

    policy_net plays BOTH sides using eps-greedy.  Because the environment
    returns states in canonical form (Channel 0 = current mover's pieces),
    the same network weights produce sensible Q-values regardless of which
    physical player is moving.

    Stores transitions with add_symmetric (horizontal-flip data augmentation).
    Applies NegaMax retroactive penalty: the move immediately before a loss
    (i.e. the opponent's last move before the winning move) gets reward -1.
    """
    state = env.reset()
    done  = False
    move_count = 0
    last_transition: dict = {}   # player_id → buffer index for retroactive penalty

    while not done:
        player = env.get_current_player()
        legal  = env.get_legal_moves()
        if not legal:
            break

        action = _select_action_eps_greedy(policy_net, state, legal, eps, device)

        next_state, reward, done = env.play_move(action)
        next_legal = env.get_legal_moves()
        next_mask  = _action_mask(next_legal)

        # Retroactive NegaMax penalty: if the current player just won,
        # the opponent's previous transition caused a loss → reward = -1.
        if done and reward == 1.0:
            opp = -player
            if opp in last_transition:
                buf.update_penalty(last_transition[opp], -1.0, True)

        buf.add_symmetric(state, action, reward, next_state, done, next_mask)
        last_transition[player] = -2   # add_symmetric adds 2 entries; original is at -2

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
    q_all  = policy_net(states_t)
    q_pred = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # Bootstrap targets from lagged target_net (NegaMax: negate opponent value)
    with torch.no_grad():
        q_next_all = target_net(next_states_t)
        q_next_all = q_next_all + (masks_t - 1) * 1e9   # mask illegal actions
        q_next_max = q_next_all.max(dim=1).values
        target_q   = rewards_t - cfg["gamma"] * q_next_max * (1.0 - dones_t)

    loss = F.mse_loss(q_pred, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_vs_random(
    policy_net: Connect4Net,
    n_games: int,
    device: str,
) -> float:
    """
    Play n_games where policy_net is the first mover (greedy, eps=0)
    vs a uniformly-random second mover.  Returns policy_net win fraction.
    Used for the periodic training progress metric.
    """
    cfg_eval = Config()
    env = ConnectFourEnvironment(cfg_eval)
    wins = 0
    policy_net.eval()
    for _ in range(n_games):
        state = env.reset()
        done  = False
        while not done:
            player = env.get_current_player()
            legal  = env.get_legal_moves()
            if not legal:
                break
            if player == cfg_eval.PLAYER_1:
                action = _select_action_eps_greedy(policy_net, state, legal, eps=0.0, device=device)
            else:
                action = random.choice(legal)
            state, reward, done = env.play_move(action)
            if done and reward == 1.0 and player == cfg_eval.PLAYER_1:
                wins += 1
    policy_net.train()
    return wins / n_games


def _evaluate_champion_bout(
    challenger: Connect4Net,
    champion: Connect4Net,
    n_games: int,
    tau: float,
    device: str,
) -> float:
    """
    Play 2 × n_games head-to-head: challenger vs champion, both sides.
    Both networks use Boltzmann sampling with temperature tau so that each
    game explores different board lines (pure argmax → only 2 unique games).

    Returns challenger win fraction over all 2 × n_games games.
    """
    cfg_eval = Config()
    env = ConnectFourEnvironment(cfg_eval)
    challenger.eval()
    champion.eval()

    wins  = 0
    total = 0

    for challenger_is_first in (True, False):
        for _ in range(n_games):
            state = env.reset()
            done  = False
            while not done:
                player = env.get_current_player()
                legal  = env.get_legal_moves()
                if not legal:
                    break
                # challenger goes first in the first half, second in the second half
                is_challenger_turn = (
                    (player == cfg_eval.PLAYER_1) == challenger_is_first
                )
                net    = challenger if is_challenger_turn else champion
                action = _select_action_boltzmann(net, state, legal, tau, device)
                state, reward, done = env.play_move(action)
                if done and reward == 1.0 and is_challenger_turn:
                    wins += 1
            total += 1

    challenger.train()
    return wins / total


# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

def _print_banner(cfg: dict, device: str) -> None:
    sep = "=" * 60
    print(sep)
    print("  Connect 4 DQN Training  [champion-challenger]")
    print(sep)
    print(f"  Device                 : {device}")
    print(f"  Total episodes         : {cfg['total_episodes']:,}")
    print(f"  Learning rate          : {cfg['lr']}")
    print(f"  Epsilon                : {cfg['eps_start']} → {cfg['eps_end']}  (decay {cfg['eps_decay']} per step)")
    print(f"  Batch size             : {cfg['batch_size']}")
    print(f"  Gamma (discount)       : {cfg['gamma']}")
    print(f"  Terminal ratio         : {cfg['terminal_ratio']:.0%}  (of each batch are terminal)")
    print(f"  Target sync every      : {cfg['target_sync_every']} gradient steps")
    print(f"  Train N times/game     : {cfg['train_n_times']}")
    print(f"  Buffer capacity        : {cfg['buffer_capacity']:,}")
    print(f"  Min buffer to train    : {cfg['min_buffer']:,}")
    print(f"  Eval vs random every   : {cfg['eval_every']} episodes  ({cfg['eval_games']} games)")
    print(f"  Dropout rate           : {cfg['dropout_rate']}")
    print(f"  Champion eval every    : {cfg['champion_eval_every']:,} episodes")
    print(f"  Champion eval games    : {cfg['champion_eval_games']} games per side  (total {2*cfg['champion_eval_games']})")
    print(f"  Champion win threshold : {cfg['champion_win_thresh']:.0%}")
    print(f"  Champion eval τ        : {cfg['champion_eval_tau']}  (Boltzmann temperature)")
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
        cfg: Override dictionary.  Keys matching DEFAULTS will replace defaults.
             Pass None to use all defaults.

    Returns:
        dict with keys: policy_net, champion_net, best_win_rate,
        final_model_path, champion_model_path, loss_history,
        win_rate_history, champion_history, total_time_s
    """
    run_cfg = {**DEFAULTS}
    if cfg:
        run_cfg.update(cfg)

    device = Config.DEVICE
    _print_banner(run_cfg, device)

    # ---- Networks -----------------------------------------------------------
    policy_net = Connect4Net(device=device, dropout_rate=run_cfg["dropout_rate"])

    # target_net: Bellman bootstrap only; lagged copy of policy_net.
    target_net = Connect4Net(device=device, dropout_rate=0.0)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # champion_net: quality gate; NOT used during training games.
    # Updated only when the challenger (policy_net) beats it in a bout.
    champion_net = Connect4Net(device=device, dropout_rate=0.0)
    champion_net.load_state_dict(policy_net.state_dict())
    champion_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=run_cfg["lr"])

    # ---- Buffer & environment -----------------------------------------------
    buf      = DQNReplayBuffer(capacity=run_cfg["buffer_capacity"])
    game_env = ConnectFourEnvironment(Config())

    # ---- Tracking -----------------------------------------------------------
    eps                  = run_cfg["eps_start"]
    total_grad_steps     = 0
    loss_window          = deque(maxlen=20)
    all_losses: List[float] = []
    win_rate_history: List[tuple] = []
    champion_history: List[dict]  = []
    unique_states: set   = set()
    best_win_rate        = 0.0
    champion_model_path: Optional[Path] = None
    n_promotions         = 0

    loss_strike         = 0
    win_collapse_strike = 0

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    hdr = f"  {'EP':>8}  {'BUF':>6}  {'TERM':>6}  {'EPS':>6}  {'LOSS(20)':>10}  {'GRAD':>7}"
    print(hdr)
    print(f"  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*7}")

    t_start = time.time()

    for episode in range(1, run_cfg["total_episodes"] + 1):

        # ---- Champion-Challenger bout ----------------------------------------
        if episode % run_cfg["champion_eval_every"] == 0:
            n_games = run_cfg["champion_eval_games"]
            t_cc    = time.time()
            cc_win  = _evaluate_champion_bout(
                policy_net, champion_net,
                n_games, run_cfg["champion_eval_tau"], device
            )
            t_cc    = time.time() - t_cc
            elapsed = time.time() - t_start

            if cc_win > run_cfg["champion_win_thresh"]:
                n_promotions += 1
                champion_net.load_state_dict(policy_net.state_dict())
                champion_net.eval()
                champion_model_path = models_dir / f"dqn_connect4_best_{timestamp}.pth"
                torch.save(policy_net.state_dict(), champion_model_path)
                champion_history.append(
                    {"episode": episode, "challenger_win": cc_win, "promoted": True}
                )
                promo_tag = f"  🏆 PROMOTED (#{n_promotions})"
            else:
                champion_history.append(
                    {"episode": episode, "challenger_win": cc_win, "promoted": False}
                )
                promo_tag = "  — challenger did not improve"

            print()
            print(
                f"  ── CHAMPION BOUT ep={episode:>7,d} ──  "
                f"challenger win={cc_win:.1%}  "
                f"(threshold {run_cfg['champion_win_thresh']:.0%})  "
                f"bout_t={t_cc:.1f}s  "
                f"total={elapsed/60:.1f}min"
                + promo_tag
            )
            print()

        # ---- Play one training game -----------------------------------------
        # policy_net plays BOTH sides, eps-greedy.
        # The canonical state flip ensures the network always acts as "the current mover".
        _play_one_game(game_env, policy_net, buf, eps, device)

        # Track unique states
        unique_states.add(game_env.get_state().tobytes())

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
                if total_grad_steps % run_cfg["target_sync_every"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())

        # ---- Compact progress line ------------------------------------------
        if episode % run_cfg["print_every"] == 0:
            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            n_term    = len(buf.terminal_buffer)
            print(
                f"  ep={episode:>8,d}  "
                f"buf={len(buf):>6,d}  "
                f"term={n_term:>5,d}  "
                f"eps={eps:.4f}  "
                f"loss(20)={mean_loss:>9.3f}  "
                f"grad={total_grad_steps:>7,d}"
            )

            if loss_window and mean_loss > LOSS_EXPLODE_THRESHOLD:
                loss_strike += 1
                print(f"  ⚠️  Loss spike #{loss_strike}/{LOSS_EXPLODE_STRIKES}  (mean={mean_loss:.1f})")
                if loss_strike >= LOSS_EXPLODE_STRIKES:
                    print(f"\n  ❌ LOSS EXPLODED — stopping at episode {episode}.")
                    break
            else:
                loss_strike = 0

        # ---- Periodic evaluation vs random ----------------------------------
        if episode % run_cfg["eval_every"] == 0:
            t_eval   = time.time()
            win_rate = _evaluate_vs_random(policy_net, run_cfg["eval_games"], device)
            t_eval   = time.time() - t_eval
            n_unique = len(unique_states)
            win_rate_history.append((episode, win_rate))

            marker = ""
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                marker = "  ← best vs random"

            mean_loss = float(np.mean(loss_window)) if loss_window else float("nan")
            elapsed   = time.time() - t_start
            print()
            print(
                f"  ── EVAL vs RANDOM ep={episode:>7,d} ──  "
                f"win={win_rate:.1%}  "
                f"unique={n_unique:,}  "
                f"loss={mean_loss:.3f}  "
                f"eval_t={t_eval:.1f}s  "
                f"total={elapsed/60:.1f}min"
                + marker
            )
            print()

            if episode > WIN_COLLAPSE_AFTER_EP and win_rate <= WIN_COLLAPSE_THRESHOLD:
                win_collapse_strike += 1
                print(f"  ⚠️  Win collapse #{win_collapse_strike}/{WIN_COLLAPSE_STRIKES}")
                if win_collapse_strike >= WIN_COLLAPSE_STRIKES:
                    print(
                        f"\n  ❌ WIN RATE COLLAPSED — stopping at episode {episode}."
                    )
                    break
            else:
                win_collapse_strike = 0

    # ---- Final model save ---------------------------------------------------
    final_path = models_dir / f"dqn_connect4_final_{timestamp}.pth"
    torch.save(
        {
            "state_dict"       : policy_net.state_dict(),
            "episode"          : episode,
            "best_win_rate"    : best_win_rate,
            "eps"              : eps,
            "total_grad_steps" : total_grad_steps,
            "hyperparams"      : run_cfg,
            "device"           : device,
            "timestamp"        : timestamp,
            "n_promotions"     : n_promotions,
            "champion_history" : champion_history,
        },
        final_path,
    )

    total_time = time.time() - t_start
    print()
    print("=" * 60)
    print(f"  Training complete in {total_time/60:.1f} min  ({total_time:.0f}s)")
    print(f"  Episodes run            : {episode:,} / {run_cfg['total_episodes']:,}")
    print(f"  Gradient steps          : {total_grad_steps:,}")
    print(f"  Best win rate vs random : {best_win_rate:.1%}")
    print(f"  Final epsilon           : {eps:.4f}")
    print(f"  Unique states seen      : {len(unique_states):,}")
    print(f"  Champion promotions     : {n_promotions}")
    if champion_history:
        print(f"  Champion bout history   :")
        for h in champion_history:
            tag = "🏆 PROMOTED" if h["promoted"] else "not promoted"
            print(f"    ep {h['episode']:>8,d}  challenger={h['challenger_win']:.1%}  {tag}")
    print(f"  Final model saved       : {final_path}")
    if champion_model_path:
        print(f"  Champion model saved    : {champion_model_path}")
    print("=" * 60)

    return dict(
        policy_net          = policy_net,
        champion_net        = champion_net,
        best_win_rate       = best_win_rate,
        final_model_path    = str(final_path),
        champion_model_path = str(champion_model_path) if champion_model_path else None,
        loss_history        = all_losses,
        win_rate_history    = win_rate_history,
        champion_history    = champion_history,
        total_time_s        = total_time,
        total_episodes      = episode,
        total_grad_steps    = total_grad_steps,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DQN agent for Connect 4")
    p.add_argument("--episodes",         type=int,   default=DEFAULTS["total_episodes"])
    p.add_argument("--lr",               type=float, default=DEFAULTS["lr"])
    p.add_argument("--eps-start",        type=float, default=DEFAULTS["eps_start"])
    p.add_argument("--eps-end",          type=float, default=DEFAULTS["eps_end"])
    p.add_argument("--eps-decay",        type=float, default=DEFAULTS["eps_decay"])
    p.add_argument("--batch-size",       type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--gamma",            type=float, default=DEFAULTS["gamma"])
    p.add_argument("--train-n",          type=int,   default=DEFAULTS["train_n_times"],
                   help="Gradient updates per episode")
    p.add_argument("--target-sync",      type=int,   default=DEFAULTS["target_sync_every"],
                   help="Gradient steps between target-net sync")
    p.add_argument("--eval-every",       type=int,   default=DEFAULTS["eval_every"])
    p.add_argument("--eval-games",       type=int,   default=DEFAULTS["eval_games"])
    p.add_argument("--buffer",           type=int,   default=DEFAULTS["buffer_capacity"])
    p.add_argument("--print-every",      type=int,   default=DEFAULTS["print_every"])
    p.add_argument("--dropout",          type=float, default=DEFAULTS["dropout_rate"])
    p.add_argument("--champion-eval",    type=int,   default=DEFAULTS["champion_eval_every"],
                   help="Episodes between champion-challenger bouts")
    p.add_argument("--champion-games",   type=int,   default=DEFAULTS["champion_eval_games"],
                   help="Games per side per champion bout")
    p.add_argument("--champion-thresh",  type=float, default=DEFAULTS["champion_win_thresh"],
                   help="Challenger win fraction required to become champion")
    p.add_argument("--champion-tau",     type=float, default=DEFAULTS["champion_eval_tau"],
                   help="Boltzmann temperature for champion bout games (0.2 = near-greedy)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg_override = dict(
        total_episodes       = args.episodes,
        lr                   = args.lr,
        eps_start            = args.eps_start,
        eps_end              = args.eps_end,
        eps_decay            = args.eps_decay,
        batch_size           = args.batch_size,
        gamma                = args.gamma,
        train_n_times        = args.train_n,
        target_sync_every    = args.target_sync,
        eval_every           = args.eval_every,
        eval_games           = args.eval_games,
        buffer_capacity      = args.buffer,
        print_every          = args.print_every,
        dropout_rate         = args.dropout,
        champion_eval_every  = args.champion_eval,
        champion_eval_games  = args.champion_games,
        champion_win_thresh  = args.champion_thresh,
        champion_eval_tau    = args.champion_tau,
    )
    train(cfg_override)
