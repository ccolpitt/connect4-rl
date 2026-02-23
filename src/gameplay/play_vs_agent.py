"""
src/gameplay/play_vs_agent.py
==============================
Interactive CLI to play Connect 4 against the trained DQN agent.

Usage
-----
    # Auto-loads the best model in models/
    python -m src.gameplay.play_vs_agent

    # Load a specific model file
    python -m src.gameplay.play_vs_agent --model models/dqn_connect4_best_20260221_232305.pth

    # You go first (default), or let the agent go first
    python -m src.gameplay.play_vs_agent --agent-first

What it shows
-------------
- Board rendered after every move (X = you, O = agent)
- Agent's Q-values for each column so you can see what it's "thinking"
- Win/loss/draw result at the end
- Option to play again

Controls
--------
    Enter column number 0-6 when prompted.
    Ctrl+C to quit.
"""

import argparse
import glob
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.environment.config import Config
from src.environment.connect4 import ConnectFourEnvironment
from src.networks.connect4_net import Connect4Net


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
DIM    = "\033[2m"


def _render_board(board: np.ndarray, last_col: Optional[int] = None) -> None:
    """Print board with colour. X=human (red), O=agent (yellow)."""
    symbols = {1: f"{RED}{BOLD}X{RESET}", -1: f"{YELLOW}{BOLD}O{RESET}", 0: "·"}
    print()
    print(f"  {DIM}┌─────────────────────┐{RESET}")
    for row in board:
        print(f"  {DIM}│{RESET} " + "  ".join(symbols[int(v)] for v in row) + f" {DIM}│{RESET}")
    print(f"  {DIM}└─────────────────────┘{RESET}")
    # Column numbers — highlight last move column
    col_labels = []
    for c in range(7):
        if c == last_col:
            col_labels.append(f"{CYAN}{BOLD}{c}{RESET}")
        else:
            col_labels.append(f"{DIM}{c}{RESET}")
    print("    " + "  ".join(col_labels))
    print()


def _render_q_values(q: np.ndarray, legal: list) -> None:
    """Print agent Q-values as a bar chart, masking illegal columns."""
    print(f"  {DIM}Agent Q-values:{RESET}")
    q_min, q_max = q.min(), q.max()
    bar_width = 20
    for col in range(7):
        if col not in legal:
            bar = f"{DIM}{'░' * bar_width}{RESET}"
            label = f"{DIM}[full]{RESET}"
        else:
            val = float(q[col])
            # Normalise to 0-1 range for bar
            if q_max > q_min:
                frac = (val - q_min) / (q_max - q_min)
            else:
                frac = 0.5
            filled = int(frac * bar_width)
            color = GREEN if val == q[legal].max() else RESET
            bar = f"{color}{'█' * filled}{'░' * (bar_width - filled)}{RESET}"
            label = f"{color}{val:+.3f}{RESET}"
        print(f"  col {col}: {bar} {label}")
    print()


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: str) -> Connect4Net:
    """Load Connect4Net weights from path. Handles both full checkpoint dicts and bare state_dicts."""
    net = Connect4Net(device=device, dropout_rate=0.0)
    raw = torch.load(model_path, map_location=device, weights_only=False)
    # train.py saves either a bare state_dict (best) or a checkpoint dict (final)
    if isinstance(raw, dict) and "state_dict" in raw:
        net.load_state_dict(raw["state_dict"])
    else:
        net.load_state_dict(raw)
    net.eval()
    return net


def _find_best_model() -> Optional[str]:
    """Return path of the most recently modified best model, or None."""
    candidates = sorted(
        glob.glob("models/dqn_connect4_best_*.pth"),
        key=os.path.getmtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Fall back to any dqn model
    candidates = sorted(
        glob.glob("models/dqn_connect4_*.pth"),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _agent_move(
    net: Connect4Net,
    state: np.ndarray,
    legal: list,
    device: str,
) -> tuple[int, np.ndarray]:
    """Return (chosen_col, q_values_array)."""
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0).to(device)
        q = net(s).squeeze(0).cpu().numpy()   # (7,)

    # Mask illegal columns
    masked = np.full(7, -1e9, dtype=np.float32)
    for m in legal:
        masked[m] = q[m]
    return int(np.argmax(masked)), q


def _human_move(legal: list) -> int:
    """Prompt until valid column entered."""
    while True:
        try:
            raw = input(f"  {BOLD}Your move{RESET} (columns: {legal}): ").strip()
            col = int(raw)
            if col in legal:
                return col
            print(f"  {RED}Column {col} is not available. Choose from {legal}.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a number 0-6.{RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            raise SystemExit(0)


# ---------------------------------------------------------------------------
# One game
# ---------------------------------------------------------------------------

def _play_game(
    net: Connect4Net,
    device: str,
    human_player: int,   # +1 or -1
    show_q: bool,
) -> str:
    """
    Play one game.  Returns 'human', 'agent', or 'draw'.
    human_player=1  → human is P1 (moves first)
    human_player=-1 → human is P2 (agent moves first)
    """
    cfg = Config()
    env = ConnectFourEnvironment(cfg)
    state = env.reset()
    done = False
    last_col: Optional[int] = None

    agent_player = -human_player

    while not done:
        current = env.get_current_player()
        legal   = env.get_legal_moves()

        _render_board(env.board, last_col)

        if current == human_player:
            # ---- Human turn ----
            if not legal:
                print("  No legal moves!")
                break
            col = _human_move(legal)
        else:
            # ---- Agent turn ----
            col, q = _agent_move(net, state, legal, device)
            if show_q:
                _render_q_values(q, legal)
            print(f"  {YELLOW}{BOLD}Agent plays column {col}{RESET}")

        state, reward, done = env.play_move(col)
        last_col = col

        if done:
            _render_board(env.board, last_col)
            mover = current   # who just moved
            if reward == 1.0:
                return "human" if mover == human_player else "agent"
            else:
                return "draw"

    return "draw"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(model_path: Optional[str] = None, agent_first: bool = False, show_q: bool = True) -> None:
    device = Config.DEVICE

    # ---- Find / load model --------------------------------------------------
    if model_path is None:
        model_path = _find_best_model()
        if model_path is None:
            print(f"{RED}No trained model found in models/. Run training first.{RESET}")
            return

    print()
    print(f"{BOLD}╔══════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║      Connect 4 vs DQN Agent          ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════╝{RESET}")
    print(f"  Model  : {model_path}")
    print(f"  Device : {device}")
    print(f"  You are: {RED}{BOLD}X{RESET}")
    print(f"  Agent  : {YELLOW}{BOLD}O{RESET}")
    print()

    net = _load_model(model_path, device)
    print(f"  {GREEN}Model loaded successfully.{RESET}")
    print()

    # ---- Game loop ----------------------------------------------------------
    wins = draws = losses = 0
    human_player = -1 if agent_first else 1

    while True:
        who_first = "Agent" if agent_first else "You"
        print(f"  {DIM}─── New game ({who_first} goes first) ───{RESET}")
        print()

        result = _play_game(net, device, human_player, show_q)

        if result == "human":
            wins += 1
            print(f"  {GREEN}{BOLD}🎉 You win!{RESET}")
        elif result == "agent":
            losses += 1
            print(f"  {YELLOW}{BOLD}🤖 Agent wins.{RESET}")
        else:
            draws += 1
            print(f"  {CYAN}{BOLD}🤝 Draw.{RESET}")

        print(f"  Score — You: {wins}  Agent: {losses}  Draw: {draws}")
        print()

        try:
            again = input("  Play again? [Y/n/s(witch first)]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if again == "s":
            agent_first = not agent_first
            human_player = -human_player
            print(f"  Switched! {'Agent' if agent_first else 'You'} now goes first.")
            print()
        elif again in ("n", "no", "q", "quit"):
            break
        # else: y / enter → play again

    print()
    print(f"  Final score — {GREEN}You: {wins}{RESET}  {YELLOW}Agent: {losses}{RESET}  Draw: {draws}")
    print("  Thanks for playing!")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play Connect 4 vs trained DQN agent")
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to model .pth file (default: newest models/dqn_connect4_best_*.pth)"
    )
    p.add_argument(
        "--agent-first", action="store_true",
        help="Let the agent move first (default: human moves first)"
    )
    p.add_argument(
        "--no-q", action="store_true",
        help="Hide agent Q-values (cleaner display)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_path  = args.model,
        agent_first = args.agent_first,
        show_q      = not args.no_q,
    )
