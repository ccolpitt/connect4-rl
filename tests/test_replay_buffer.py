"""
Unit tests for DQNReplayBuffer (src/utils/dqn_replay_buffer.py).

Phase 3 tests: structural / mechanical correctness of the buffer class.
Learning convergence is tested separately in tests/test_synthetic_learning.py.

"terminal_ratio" explained:
  In Connect 4, most transitions are non-terminal (reward=0, done=False).
  Terminal transitions (win reward=+1 or loss reward=-1, done=True) are rare,
  maybe 1 in 15-20 entries. Uniform sampling means the network rarely sees
  win/loss signals → slow convergence.
  Fix: DQNReplayBuffer keeps a separate terminal_buffer deque.
  sample(batch_size, terminal_ratio=0.30) pulls 30% from terminal_buffer and
  70% from the main buffer, then shuffles. Tests verify this fraction holds.

Covers:
  - add(): storage, len, capacity eviction, terminal routing
  - add_symmetric(): 2 entries, action mirrored, state/mask flipped
  - update_penalty(): reward/done override, terminal_buffer append, no-op on bad index
  - sample(indices=...): deterministic retrieval
  - sample(terminal_ratio=...): balanced batches, correct dtypes/shapes
  - is_ready(), clear(), get_stats(), __repr__
"""

import pytest
import numpy as np

from utils.dqn_replay_buffer import DQNReplayBuffer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state():
    return np.random.rand(2, 6, 7).astype(np.float32)

def _legal():
    return np.ones(7, dtype=np.float32)

def _done_mask():
    return np.zeros(7, dtype=np.float32)

def _add_n(buf, n, terminal=False):
    for i in range(n):
        buf.add(_state(), i % 7, 1.0 if terminal else 0.0,
                _state(), terminal, _done_mask() if terminal else _legal())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def buf():
    return DQNReplayBuffer(capacity=100)

@pytest.fixture
def buf_mixed():
    """20 non-terminal + 10 terminal."""
    b = DQNReplayBuffer(capacity=100)
    _add_n(b, 20, terminal=False)
    _add_n(b, 10, terminal=True)
    return b


# ---------------------------------------------------------------------------
# 1. add
# ---------------------------------------------------------------------------

class TestAdd:

    def test_empty_len_zero(self, buf):
        assert len(buf) == 0

    def test_add_increments_len(self, buf):
        _add_n(buf, 5)
        assert len(buf) == 5

    def test_capacity_eviction(self):
        b = DQNReplayBuffer(capacity=10)
        _add_n(b, 20)
        assert len(b) == 10

    def test_stored_action(self, buf):
        buf.add(_state(), 5, 0.0, _state(), False, _legal())
        assert buf.buffer[0].action == 5

    def test_stored_reward(self, buf):
        buf.add(_state(), 3, 1.0, _state(), True, _done_mask())
        assert buf.buffer[0].reward == 1.0

    def test_terminal_routed_to_terminal_buffer(self, buf):
        buf.add(_state(), 3, 1.0, _state(), True, _done_mask())
        assert len(buf.terminal_buffer) == 1

    def test_non_terminal_not_in_terminal_buffer(self, buf):
        buf.add(_state(), 3, 0.0, _state(), False, _legal())
        assert len(buf.terminal_buffer) == 0


# ---------------------------------------------------------------------------
# 2. add_symmetric
# ---------------------------------------------------------------------------

class TestAddSymmetric:

    def test_adds_two_entries(self, buf):
        buf.add_symmetric(_state(), 2, 0.0, _state(), False, _legal())
        assert len(buf) == 2

    def test_original_action_intact(self, buf):
        buf.add_symmetric(_state(), 2, 0.0, _state(), False, _legal())
        assert buf.buffer[0].action == 2

    def test_mirrored_action(self, buf):
        buf.add_symmetric(_state(), 2, 0.0, _state(), False, _legal())
        assert buf.buffer[1].action == 4  # 6 - 2

    @pytest.mark.parametrize("col,expected", [(0, 6), (1, 5), (3, 3), (6, 0)])
    def test_mirror_formula(self, buf, col, expected):
        buf.add_symmetric(_state(), col, 0.0, _state(), False, _legal())
        assert buf.buffer[1].action == expected

    def test_state_flipped_horizontally(self, buf):
        s = np.zeros((2, 6, 7), dtype=np.float32)
        s[0, 5, 0] = 1.0   # piece at leftmost column
        buf.add_symmetric(s, 0, 0.0, _state(), False, _legal())
        mirrored = buf.buffer[1].state
        assert mirrored[0, 5, 6] == 1.0   # appears at rightmost column
        assert mirrored[0, 5, 0] == 0.0

    def test_mask_flipped(self, buf):
        mask = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        buf.add_symmetric(_state(), 0, 0.0, _state(), False, mask)
        expected = np.array([0, 0, 0, 0, 0, 1, 1], dtype=np.float32)
        np.testing.assert_array_equal(buf.buffer[1].next_mask, expected)

    def test_terminal_both_go_to_terminal_buffer(self, buf):
        buf.add_symmetric(_state(), 3, 1.0, _state(), True, _done_mask())
        assert len(buf.terminal_buffer) == 2


# ---------------------------------------------------------------------------
# 3. update_penalty
# ---------------------------------------------------------------------------

class TestUpdatePenalty:

    def test_updates_last_entry(self, buf):
        _add_n(buf, 5)
        buf.update_penalty(-1, -1.0, True)
        assert buf.buffer[-1].reward == -1.0
        assert buf.buffer[-1].done == True

    def test_updates_second_to_last(self, buf):
        _add_n(buf, 5)
        buf.update_penalty(-2, -1.0, True)
        assert buf.buffer[-2].reward == -1.0

    def test_other_entries_unchanged(self, buf):
        _add_n(buf, 5)
        r0 = buf.buffer[0].reward
        buf.update_penalty(-1, -1.0, True)
        assert buf.buffer[0].reward == r0

    def test_updated_entry_to_terminal_buffer(self, buf):
        _add_n(buf, 5)
        before = len(buf.terminal_buffer)
        buf.update_penalty(-1, -1.0, True)
        assert len(buf.terminal_buffer) == before + 1

    def test_out_of_range_no_crash(self, buf):
        _add_n(buf, 3)
        buf.update_penalty(-99, -1.0, True)   # silent no-op
        assert len(buf) == 3

    def test_second_to_last_fix_pattern(self, buf):
        """
        Reproduces the training-loop idiom:
          add_symmetric (loser move) → 2 entries  (-4, -3)
          add_symmetric (winner move) → 2 entries  (-2, -1)
          update_penalty(-3) and (-4) → loser move gets reward=-1
        """
        buf.add_symmetric(_state(), 3, 0.0, _state(), False, _legal())  # loser's move
        buf.add_symmetric(_state(), 4, 1.0, _state(), True, _done_mask())  # winner's move
        buf.update_penalty(-3, -1.0, True)
        buf.update_penalty(-4, -1.0, True)
        assert buf.buffer[-3].reward == -1.0
        assert buf.buffer[-4].reward == -1.0
        assert buf.buffer[-1].reward == 1.0   # winner untouched
        assert buf.buffer[-2].reward == 1.0


# ---------------------------------------------------------------------------
# 4. sample with explicit indices
# ---------------------------------------------------------------------------

class TestSampleByIndex:

    def test_specific_index_retrieval(self, buf):
        _add_n(buf, 10)
        buf.buffer[3] = buf.buffer[3]._replace(reward=99.0)
        _, _, r, _, _, _ = buf.sample(1, indices=[3])
        assert r[0] == 99.0

    def test_negative_index(self, buf):
        _add_n(buf, 5)
        # last action is (5-1) % 7 == 4
        _, a, _, _, _, _ = buf.sample(1, indices=[-1])
        assert a[0] == 4

    def test_shapes_from_index_sample(self, buf):
        _add_n(buf, 10)
        s, a, r, ns, d, m = buf.sample(3, indices=[0, 1, 2])
        assert s.shape == (3, 2, 6, 7)
        assert a.shape == (3,)
        assert r.shape == (3,)
        assert ns.shape == (3, 2, 6, 7)
        assert d.shape == (3,)
        assert m.shape == (3, 7)


# ---------------------------------------------------------------------------
# 5. sample with terminal_ratio (balanced)
# ---------------------------------------------------------------------------

class TestSampleBalanced:

    def test_raises_if_too_few(self, buf):
        _add_n(buf, 5)
        with pytest.raises(ValueError):
            buf.sample(10)

    def test_correct_batch_size(self, buf_mixed):
        _, a, _, _, _, _ = buf_mixed.sample(16)
        assert len(a) == 16

    def test_terminal_ratio_approximately_correct(self, buf_mixed):
        """
        With terminal_ratio=0.30 and buf_mixed (20 non-terminal + 10 terminal = 30 total):
          - n_terminals forced = int(16 * 0.30) = 4
          - remaining 12 sampled from main buffer (10/30 ≈ 33% terminal)
          - expected terminal from main = 12 * 10/30 = 4
          - total expected terminal fraction ≈ 8/16 = 0.50

        The forced ratio is a MINIMUM guarantee, not the exact fraction in the batch,
        because terminal_buffer entries also live in the main buffer.
        Test that the fraction is between 0.30 (lower bound guarantee) and 0.70.
        """
        fracs = [buf_mixed.sample(16, terminal_ratio=0.30)[4].mean()
                 for _ in range(100)]
        avg = float(np.mean(fracs))
        assert 0.25 < avg < 0.70, f"Terminal ratio off: {avg:.3f}"

    def test_fallback_no_terminals(self):
        b = DQNReplayBuffer(capacity=100)
        _add_n(b, 20, terminal=False)
        s, a, r, ns, d, m = b.sample(16, terminal_ratio=0.5)
        assert len(a) == 16

    def test_array_dtypes(self, buf_mixed):
        s, a, r, ns, d, m = buf_mixed.sample(8)
        assert s.dtype == np.float32
        assert r.dtype == np.float32
        assert d.dtype == np.float32
        assert a.dtype == np.int64    # must be int64 for torch.gather()


# ---------------------------------------------------------------------------
# 6. Helpers: is_ready, clear, get_stats, repr
# ---------------------------------------------------------------------------

class TestHelpers:

    def test_is_ready_false(self, buf):
        _add_n(buf, 5)
        assert not buf.is_ready(10)

    def test_is_ready_true_at_threshold(self, buf):
        _add_n(buf, 10)
        assert buf.is_ready(10)

    def test_clear_empties(self, buf):
        _add_n(buf, 10)
        buf.clear()
        assert len(buf) == 0

    def test_get_stats_keys(self, buf):
        _add_n(buf, 5)
        for k in ("size", "capacity", "utilization", "is_full"):
            assert k in buf.get_stats()

    def test_get_stats_values(self, buf):
        _add_n(buf, 5)
        s = buf.get_stats()
        assert s["size"] == 5
        assert s["capacity"] == 100
        assert abs(s["utilization"] - 0.05) < 1e-9
        assert s["is_full"] == False

    def test_full_buffer_flag(self):
        b = DQNReplayBuffer(capacity=5)
        _add_n(b, 5)
        assert b.get_stats()["is_full"] == True

    def test_repr(self, buf):
        _add_n(buf, 7)
        r = repr(buf)
        assert "DQNReplayBuffer" in r
        assert "7" in r


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
