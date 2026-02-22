# Connect4-RL: DQN Training Plan & Status

## How to Resume After Context Window Reset
1. Read this file.
2. Run `pytest` — passing tests = what's done; failures = what needs fixing.
3. Skim the "Key Design Decisions" section below.
4. Continue from the first `[ ]` phase below.

---

## Project Goal
Train a DQN agent to master Connect 4 using a convolutional network.
End state: agent beats a random opponent >90% of the time, and a human can play against it.

---

## Phase Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 | ✅ Done | Project briefing & plan |
| 1 | ✅ Done | pytest harness: `conftest.py`, `pytest.ini`, fixed `tests/test_environment.py` (32 tests) |
| 2 | ✅ Done | `src/networks/connect4_net.py` (extracted from train_dqn_20251221.py), `tests/test_network.py` (24 tests) |
| 3 | 🔲 Next | `tests/test_replay_buffer.py` — unit tests for DQNReplayBuffer |
| 4 | 🔲 Todo | `tests/test_synthetic_learning.py` — idealized buffer → Q values converge ±1 |
| 5 | 🔲 Todo | `tests/test_training_mechanics.py` — ~300 episode smoke test |
| 6 | 🔲 Todo | `src/training/train.py` — full 10k-50k episode training script |
| 7 | 🔲 Todo | `src/gameplay/play_vs_agent.py` — interactive CLI to play vs trained model |
| 8 | 🔲 Todo | Iterate arch/hyperparams if win rate stalls below 70% vs random |

---

## Key Design Decisions

### State Representation
- Shape: `(2, 6, 7)` — **canonical 2-channel**
  - Channel 0: current player's pieces ("MY pieces")
  - Channel 1: opponent's pieces
- **Always from the current player's perspective** — env.get_state() auto-flips
- This means one network handles both players (same weights, same perspective)

### Reward Convention (from `env.play_move()`)
- `+1.0` — moving player wins
- `0.0`  — game continues OR draw
- Moving player can never get `-1.0` directly from play_move
- **Second-to-last move fix**: after a win, the opponent's prior move in the replay buffer gets `reward=-1.0, done=True` via `replay_buffer.update_penalty(index=-3, new_reward=-1.0, is_done=True)` (index=-3 because `add_symmetric` adds 2 entries per move)

### Bellman Target (NegaMax-style)
```
target_q = reward - gamma * max(Q_target(next_state)) * (1 - done)
```
Note the **subtraction** (not addition). This is because after the current player moves, the next state is seen from the *opponent's* perspective — a good state for the opponent is bad for us.

### Symmetric Augmentation
`replay_buffer.add_symmetric()` stores both the original transition AND a horizontally-mirrored version (column `a` → column `6-a`). This doubles the effective training data.

### Terminal Ratio Sampling
`replay_buffer.sample(batch_size, terminal_ratio=0.3)` ensures 30% of each batch contains terminal states (wins/losses). This prevents the imbalanced buffer problem where terminal states are rare.

### Policy vs Target Network
- **Policy net**: trained every step via gradient descent
- **Target net**: frozen copy, synced every `TARGET_UPDATE_FREQ=100` episodes via `load_state_dict`
- Target net produces the Bellman targets — this stabilises training

### Architecture (Connect4Net — `src/networks/connect4_net.py`)
```
Input (B, 2, 6, 7)
→ Conv1: 2→64, 3×3, pad=1 + BN + ReLU + Dropout2d
→ Conv2: 64→64, 3×3, pad=1 + BN + ReLU + Dropout2d
→ Conv3: 64→64, 3×3, pad=1 + BN + ReLU + Dropout2d
→ Flatten → 2688
→ FC: 2688→128 + ReLU + Dropout
→ Output: 128→7  (raw Q-values, no activation)
```
He (Kaiming) init for all weights. ~500k parameters.

### Config Defaults (`src/environment/config.py`)
- `NUM_EPISODES = 500` (increase to 10k-50k for real training)
- `EPS_START = 0.5`, `EPS_END = 0.2`, `EPS_DECAY = 0.9999`
- `BATCH_SIZE = 128`, `TRAIN_N_TIMES_PER_GAME = 4`
- `GAMMA = 0.99`, `TARGET_UPDATE_FREQ = 100`
- `TERMINAL_RATE = 0.3` (30% terminal states per batch)
- `DROPOUT_RATE = 0.00` (use 0.1 for training if overfitting)

---

## Key Files
| File | Purpose |
|------|---------|
| `src/environment/connect4.py` | Game logic, canonical state, reward |
| `src/environment/config.py` | All hyperparameters |
| `src/networks/connect4_net.py` | Q-network (the working architecture) |
| `src/utils/dqn_replay_buffer.py` | Replay buffer with terminal ratio + symmetric add |
| `notebooks/training_examples_last_2_moves_20251221.py` | Synthetic replay buffer generator |
| `src/training/train_dqn_20251221.py` | Original monolithic training script (reference) |
| `tests/test_environment.py` | 32 env tests ✅ |
| `tests/test_network.py` | 24 network tests ✅ |
| `conftest.py` | sys.path setup for pytest |
| `pytest.ini` | pytest configuration |

---

## Running Tests
```bash
# All tests
pytest

# Single file
pytest tests/test_environment.py -v
pytest tests/test_network.py -v

# With output (for debugging)
pytest tests/test_synthetic_learning.py -v -s
```

---

## If Win Rate Stalls (Phase 8 levers)
1. Increase `EPS_START` to 0.8, slow `EPS_DECAY` to 0.99999
2. Increase `NUM_EPISODES` to 50k+
3. Increase `TARGET_UPDATE_FREQ` to 500 (more stable targets)
4. Tune `learning_rate`: try 1e-4 (current 1e-5 may be too slow)
5. Deeper network: add a 4th conv layer or wider FC (256)
6. Prioritized experience replay (already in `src/utils/prioritized_replay_buffer.py`)
7. Only then: consider A2C/PPO
