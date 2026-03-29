"""
Play Connect 4 — human vs agent, or agent vs agent.

Architecture-agnostic: loads TorchScript (.pt) models via torch.jit.load.
No knowledge of the network class is needed.

Usage:
    # Human vs current champion
    python play_champion.py

    # Human vs specific champion
    python play_champion.py --p2 models/champion_v3_20260328.pt

    # Agent goes first
    python play_champion.py --agent-first

    # Agent vs agent (watch two policies play)
    python play_champion.py --p1 models/champion_v2.pt --p2 models/champion_v5.pt

    # Hide Q-values
    python play_champion.py --no-q
"""
import sys, os, argparse, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.environment import ConnectFourEnvironment, Config

# Terminal colors
RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
DIM    = "\033[2m"


def load_policy(path, device="cpu"):
    """Load a TorchScript policy on CPU. Returns (model, metadata_str)."""
    extra = {"metadata.txt": ""}
    model = torch.jit.load(path, map_location="cpu", _extra_files=extra)
    model.eval()
    meta = extra["metadata.txt"]
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8")
    return model, meta


def agent_move(model, state, legal, device="cpu"):
    """Greedy action from a policy model. Returns (action, q_values)."""
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0)
        q = model(s).squeeze(0).cpu().numpy()
    masked = np.full(7, -1e9, dtype=np.float32)
    for m in legal:
        masked[m] = q[m]
    return int(np.argmax(masked)), q


def render_board(board, last_col=None):
    symbols = {1: f"{RED}{BOLD}X{RESET}", -1: f"{YELLOW}{BOLD}O{RESET}", 0: "·"}
    print(f"\n  {DIM}┌─────────────────────┐{RESET}")
    for row in board:
        print(f"  {DIM}│{RESET} " + "  ".join(symbols[int(v)] for v in row) + f" {DIM}│{RESET}")
    print(f"  {DIM}└─────────────────────┘{RESET}")
    cols = [f"{CYAN}{BOLD}{c}{RESET}" if c == last_col else f"{DIM}{c}{RESET}" for c in range(7)]
    print("    " + "  ".join(cols) + "\n")


def render_q(q, legal, label="Agent"):
    print(f"  {DIM}{label} Q-values:{RESET}")
    best_val = max(q[c] for c in legal) if legal else 0
    for col in range(7):
        if col not in legal:
            print(f"  col {col}: {DIM}{'░' * 20} [full]{RESET}")
        else:
            val = float(q[col])
            bar_len = int(max(0, min(20, (val + 1) * 10)))
            color = GREEN if val == best_val else RESET
            print(f"  col {col}: {color}{'█' * bar_len}{'░' * (20 - bar_len)} {val:+.3f}{RESET}")
    print()


def human_move(legal):
    while True:
        try:
            raw = input(f"  {BOLD}Your move{RESET} (columns: {legal}): ").strip()
            col = int(raw)
            if col in legal:
                return col
            print(f"  {RED}Column {col} not available. Choose from {legal}.{RESET}")
        except ValueError:
            print(f"  {RED}Enter a number 0-6.{RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            raise SystemExit(0)


def play_game(p1, p2, device, show_q=True):
    """Play one game. p1/p2 are either a loaded model or 'human'.
    Returns 'p1', 'p2', or 'draw'."""
    env = ConnectFourEnvironment(Config())
    env.reset()
    done = False
    last_col = None
    moves = 0

    players = [p1, p2]  # p1 is Player 1 (+1), p2 is Player 2 (-1)
    labels = ["P1", "P2"]
    colors = [RED, YELLOW]
    symbols = ["X", "O"]

    while not done and moves < 42:
        idx = moves % 2  # 0 = P1's turn, 1 = P2's turn
        player = players[idx]
        state = env.get_state()
        legal = env.get_legal_moves()
        render_board(env.board, last_col)

        if player == "human":
            col = human_move(legal)
        else:
            col, q = agent_move(player, state, legal, device)
            if show_q:
                render_q(q, legal, label=f"{labels[idx]} ({symbols[idx]})")
            print(f"  {colors[idx]}{BOLD}{labels[idx]} plays column {col}{RESET}")

        _, reward, done = env.play_move(col)
        last_col = col
        moves += 1

        if done:
            render_board(env.board, last_col)
            if reward == 1.0:
                winner_idx = (moves - 1) % 2
                return "p1" if winner_idx == 0 else "p2"
            return "draw"
    return "draw"


def find_current_champion():
    """Find the current champion .pt file."""
    current = "models/champion_current.pt"
    if os.path.exists(current):
        return current
    # Fallback: newest champion_v*.pt
    candidates = sorted(glob.glob("models/champion_v*.pt"), key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def list_policies():
    """List all saved champion policies with metadata."""
    files = sorted(glob.glob("models/champion_v*.pt"), key=os.path.getmtime)
    if not files:
        print(f"  {RED}No champion policies found in models/{RESET}")
        return []
    
    print(f"\n  {BOLD}Available Champion Policies:{RESET}")
    print(f"  {'─' * 60}")
    for f in files:
        extra = {"metadata.txt": ""}
        try:
            torch.jit.load(f, map_location="cpu", _extra_files=extra)
            meta = extra["metadata.txt"]
            if isinstance(meta, bytes):
                meta = meta.decode("utf-8")
            # Parse metadata
            info = {}
            for line in meta.strip().split("\n"):
                if "=" in line:
                    k, v = line.split("=", 1)
                    info[k.strip()] = v.strip()
            ver = info.get("version", "?")
            ep = info.get("episode", "?")
            wr = info.get("win_rate_vs_champion", "?")
            ts = info.get("timestamp", "?")
            print(f"  v{ver:>3s}  ep {ep:>6s}  wr {wr:>5s}  {ts}  {DIM}{os.path.basename(f)}{RESET}")
        except Exception:
            print(f"  {DIM}{os.path.basename(f)} (could not read metadata){RESET}")
    print(f"  {'─' * 60}")
    
    # Show current champion
    current = "models/champion_current.pt"
    if os.path.exists(current):
        print(f"  Current champion: {CYAN}champion_current.pt{RESET}")
    print()
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Play Connect 4 — human vs agent or agent vs agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python play_champion.py                          # Human vs current champion
  python play_champion.py --agent-first            # Agent moves first
  python play_champion.py --p2 models/champion_v3.pt  # Human vs specific policy
  python play_champion.py --p1 models/v2.pt --p2 models/v5.pt  # Agent vs agent
  python play_champion.py --list                   # List all saved policies
""")
    parser.add_argument("--p1", type=str, default=None,
                        help="P1 policy .pt file (default: human)")
    parser.add_argument("--p2", type=str, default=None,
                        help="P2 policy .pt file (default: current champion)")
    parser.add_argument("--agent-first", action="store_true",
                        help="Agent plays P1 (moves first). Human plays P2.")
    parser.add_argument("--no-q", action="store_true",
                        help="Hide Q-value display")
    parser.add_argument("--list", action="store_true",
                        help="List all saved champion policies and exit")
    parser.add_argument("--games", type=int, default=0,
                        help="For agent-vs-agent: number of games (0 = interactive)")
    args = parser.parse_args()

    device = str(Config().DEVICE)

    if args.list:
        list_policies()
        return

    # Determine who is P1 and P2
    # Default: human vs current champion
    p1_model = None
    p2_model = None
    p1_label = "Human"
    p2_label = "Human"

    if args.p1 and args.p2:
        # Agent vs agent
        p1_model, p1_meta = load_policy(args.p1, device)
        p2_model, p2_meta = load_policy(args.p2, device)
        p1_label = os.path.basename(args.p1)
        p2_label = os.path.basename(args.p2)
    elif args.agent_first:
        # Agent is P1, human is P2
        model_path = args.p2 or args.p1 or find_current_champion()
        if not model_path:
            print(f"{RED}No champion found. Train first.{RESET}")
            return
        p1_model, p1_meta = load_policy(model_path, device)
        p1_label = os.path.basename(model_path)
        p2_label = "Human"
    else:
        # Human is P1, agent is P2
        model_path = args.p2 or args.p1 or find_current_champion()
        if not model_path:
            print(f"{RED}No champion found. Train first.{RESET}")
            return
        p2_model, p2_meta = load_policy(model_path, device)
        p1_label = "Human"
        p2_label = os.path.basename(model_path)

    p1 = p1_model if p1_model else "human"
    p2 = p2_model if p2_model else "human"

    print(f"\n{BOLD}╔══════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║         Connect 4 Arena               ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════╝{RESET}")
    print(f"  P1 ({RED}X{RESET}): {p1_label}")
    print(f"  P2 ({YELLOW}O{RESET}): {p2_label}")
    print(f"  Device: {device}\n")

    # Agent vs agent batch mode
    if p1 != "human" and p2 != "human" and args.games > 0:
        p1_wins = p2_wins = draws = 0
        for g in range(args.games):
            result = play_game(p1, p2, device, show_q=False)
            if result == "p1":
                p1_wins += 1
            elif result == "p2":
                p2_wins += 1
            else:
                draws += 1
        print(f"\n  {BOLD}Results ({args.games} games):{RESET}")
        print(f"  {RED}P1{RESET} ({p1_label}): {p1_wins} wins ({100*p1_wins/args.games:.0f}%)")
        print(f"  {YELLOW}P2{RESET} ({p2_label}): {p2_wins} wins ({100*p2_wins/args.games:.0f}%)")
        print(f"  Draws: {draws}\n")
        return

    # Interactive mode
    p1_wins = p2_wins = draws = 0
    while True:
        print(f"  {DIM}─── New game ───{RESET}\n")
        result = play_game(p1, p2, device, show_q=not args.no_q)

        if result == "p1":
            p1_wins += 1
            winner = p1_label
            print(f"  {GREEN}{BOLD}{winner} wins!{RESET}")
        elif result == "p2":
            p2_wins += 1
            winner = p2_label
            print(f"  {GREEN}{BOLD}{winner} wins!{RESET}")
        else:
            draws += 1
            print(f"  {CYAN}{BOLD}Draw.{RESET}")

        total = p1_wins + p2_wins + draws
        print(f"  Score — {p1_label}: {p1_wins}  {p2_label}: {p2_wins}  Draws: {draws}\n")

        # If both are agents, auto-continue or stop
        if p1 != "human" and p2 != "human":
            try:
                again = input("  Continue? [Y/n]: ").strip().lower()
                if again in ("n", "no", "q"):
                    break
            except (EOFError, KeyboardInterrupt):
                break
        else:
            try:
                again = input("  Play again? [Y/n/s(witch sides)]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break
            if again == "s":
                p1, p2 = p2, p1
                p1_label, p2_label = p2_label, p1_label
                p1_wins, p2_wins = p2_wins, p1_wins
                print(f"  Switched! P1={p1_label}, P2={p2_label}\n")
            elif again in ("n", "no", "q", "quit"):
                break

    print(f"\n  Final — {p1_label}: {p1_wins}  {p2_label}: {p2_wins}  Draws: {draws}")
    print("  Thanks for playing!\n")


if __name__ == "__main__":
    main()
