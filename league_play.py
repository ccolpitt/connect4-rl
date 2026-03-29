"""
League Play — Round-robin tournament between champion policies.

Architecture-agnostic: loads TorchScript (.pt) models.

Usage:
    # Auto-find all champions and run tournament
    python league_play.py

    # Specific policies
    python league_play.py --policies models/champion_v1.pt models/champion_v3.pt models/champion_v5.pt

    # More games per matchup
    python league_play.py --games 50

    # Only use the last N champions
    python league_play.py --last 8
"""
import sys, os, argparse, glob, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.environment import ConnectFourEnvironment, Config


def load_policy(path):
    """Load TorchScript policy on CPU."""
    extra = {"metadata.txt": ""}
    model = torch.jit.load(path, map_location="cpu", _extra_files=extra)
    model.eval()
    meta = extra["metadata.txt"]
    if isinstance(meta, bytes):
        meta = meta.decode("utf-8")
    return model, meta


def agent_move(model, state, legal):
    """Greedy action selection."""
    with torch.no_grad():
        s = torch.from_numpy(state).float().unsqueeze(0)
        q = model(s).squeeze(0).cpu().numpy()
    masked = np.full(7, -1e9, dtype=np.float32)
    for m in legal:
        masked[m] = q[m]
    return int(np.argmax(masked))


def play_match(model_a, model_b, num_games=20):
    """Play num_games between two policies. Returns (a_wins, b_wins, draws)."""
    env = ConnectFourEnvironment(Config())
    a_wins = b_wins = draws = 0

    for g in range(num_games):
        env.reset()
        done = False
        moves = 0
        # Alternate who goes first
        a_is_p1 = (g % 2 == 0)
        reward = 0.0

        while not done and moves < 42:
            state = env.get_state()
            legal = env.get_legal_moves()
            is_p1_turn = (moves % 2 == 0)

            if (is_p1_turn and a_is_p1) or (not is_p1_turn and not a_is_p1):
                action = agent_move(model_a, state, legal)
            else:
                action = agent_move(model_b, state, legal)

            _, reward, done = env.play_move(action)
            moves += 1

        if reward == 1.0:
            last_was_p1 = ((moves - 1) % 2 == 0)
            a_made_last = (last_was_p1 == a_is_p1)
            if a_made_last:
                a_wins += 1
            else:
                b_wins += 1
        else:
            draws += 1

    return a_wins, b_wins, draws


def parse_metadata(meta_str):
    """Parse metadata string into dict."""
    info = {}
    for line in meta_str.strip().split("\n"):
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return info


def find_champions(last_n=None):
    """Find champion .pt files, optionally only the last N."""
    files = sorted(glob.glob("models/champion_v*.pt"), key=os.path.getmtime)
    # Exclude v0 (untrained) and v999 (test artifacts)
    files = [f for f in files if "v0_" not in os.path.basename(f) and "v999" not in os.path.basename(f)]
    if last_n and len(files) > last_n:
        files = files[-last_n:]
    return files


def run_tournament(policy_paths, games_per_matchup=20):
    """Run round-robin tournament. Returns win matrix and labels."""
    n = len(policy_paths)
    labels = []
    models = []

    print(f"\nLoading {n} policies...")
    for path in policy_paths:
        model, meta = load_policy(path)
        info = parse_metadata(meta)
        ver = info.get("version", "?")
        ep = info.get("episode", "?")
        label = f"v{ver} (ep{ep})"
        labels.append(label)
        models.append(model)
        print(f"  {label}: {os.path.basename(path)}")

    # Win matrix: wins[i][j] = how many times i beat j
    wins = np.zeros((n, n), dtype=int)
    draws_mat = np.zeros((n, n), dtype=int)

    total_matchups = n * (n - 1) // 2
    print(f"\nRunning {total_matchups} matchups ({games_per_matchup} games each)...\n")

    matchup = 0
    for i, j in itertools.combinations(range(n), 2):
        matchup += 1
        a_wins, b_wins, d = play_match(models[i], models[j], games_per_matchup)
        wins[i][j] = a_wins
        wins[j][i] = b_wins
        draws_mat[i][j] = d
        draws_mat[j][i] = d
        print(f"  [{matchup}/{total_matchups}] {labels[i]} vs {labels[j]}: "
              f"{a_wins}-{b_wins}-{d}")

    # Compute rankings
    total_wins = wins.sum(axis=1)
    total_losses = wins.sum(axis=0)
    total_draws_per = draws_mat.sum(axis=1) // 2 if n > 1 else np.zeros(n)
    total_games = total_wins + total_losses + total_draws_per
    win_pct = np.where(total_games > 0, total_wins / total_games, 0)

    # Sort by win percentage
    ranking = np.argsort(-win_pct)

    print(f"\n{'='*60}")
    print(f"  LEAGUE STANDINGS")
    print(f"{'='*60}")
    print(f"  {'Rank':<6}{'Policy':<20}{'W':<6}{'L':<6}{'D':<6}{'Win%':<8}")
    print(f"  {'-'*52}")
    for rank, idx in enumerate(ranking):
        print(f"  {rank+1:<6}{labels[idx]:<20}{total_wins[idx]:<6}{total_losses[idx]:<6}"
              f"{int(total_draws_per[idx]):<6}{win_pct[idx]:.1%}")
    print(f"{'='*60}\n")

    return wins, labels, ranking, win_pct


def main():
    parser = argparse.ArgumentParser(description="League play between champion policies")
    parser.add_argument("--policies", nargs="+", type=str, default=None,
                        help="Specific .pt files to include")
    parser.add_argument("--games", type=int, default=20,
                        help="Games per matchup (default: 20)")
    parser.add_argument("--last", type=int, default=None,
                        help="Only use the last N champions")
    args = parser.parse_args()

    if args.policies:
        paths = args.policies
    else:
        paths = find_champions(last_n=args.last)

    if len(paths) < 2:
        print("Need at least 2 policies for a tournament.")
        print("Run training first, or specify --policies.")
        return

    run_tournament(paths, games_per_matchup=args.games)


if __name__ == "__main__":
    main()
