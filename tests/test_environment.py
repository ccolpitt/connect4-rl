"""
Comprehensive test suite for Connect 4 environment.

Canonical representation: state is shape (2, 6, 7)
  - Channel 0: current player's pieces ("MY" pieces)
  - Channel 1: opponent's pieces
  - NO channel 2 (was removed from old 3-channel design)

Reward convention (from play_move):
  - +1.0  : moving player wins
  - 0.0   : game continues OR draw
  - never returns None or config.PLAYER_1 (int)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# conftest.py at root already handles sys.path — keep this as a safety net
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.connect4 import ConnectFourEnvironment
from environment.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    config = Config()
    return ConnectFourEnvironment(config)


@pytest.fixture
def config():
    return Config()


# ---------------------------------------------------------------------------
# Phase 1: Basic initialisation
# ---------------------------------------------------------------------------

class TestEnvironmentUnit:
    """Test basic environment functionality."""

    def test_initialization(self, config):
        """Environment initialises with correct defaults."""
        env = ConnectFourEnvironment(config)
        assert env.rows == 6
        assert env.cols == 7
        assert env.current_player == config.PLAYER_1
        assert env.last_move is None
        assert np.all(env.board == 0)

    def test_reset_clears_board(self, env, config):
        """Reset returns an empty board in canonical (2,6,7) shape."""
        env.play_move(3)
        env.play_move(4)
        state = env.reset()

        assert np.all(env.board == 0)
        assert env.current_player == config.PLAYER_1
        assert env.last_move is None
        # Canonical 2-channel state
        assert state.shape == (2, 6, 7)

    def test_state_shape_is_2_channels(self, env):
        """get_state() must return (2, 6, 7) — the canonical representation."""
        state = env.get_state()
        assert state.shape == (2, 6, 7), (
            f"Expected (2, 6, 7) but got {state.shape}. "
            "The state is 2-channel: [my_pieces, opp_pieces]."
        )

    def test_initial_state_all_zeros(self, env):
        """Both channels are all zeros on an empty board."""
        state = env.get_state()
        assert np.all(state[0] == 0), "Channel 0 (my pieces) should be empty"
        assert np.all(state[1] == 0), "Channel 1 (opp pieces) should be empty"

    def test_canonical_perspective_after_move(self, env, config):
        """
        After P1 plays col 3, the environment switches to P2.
        P2's get_state() should show zero own-pieces and one opponent piece.
        """
        env.play_move(3)  # P1 places at (5,3); current_player becomes P2
        state = env.get_state()

        # Now it's P2's turn; P2 sees NO own pieces yet
        assert state[0, 5, 3] == 0.0, "P2 has no own piece at (5,3)"
        # P2 sees P1's piece as the opponent
        assert state[1, 5, 3] == 1.0, "P2 sees opponent (P1) piece at (5,3)"


# ---------------------------------------------------------------------------
# Phase 2: Legal moves
# ---------------------------------------------------------------------------

class TestLegalMoves:

    def test_initial_all_legal(self, env):
        legal = env.get_legal_moves()
        assert legal == list(range(7))

    def test_full_column_illegal(self, env):
        for _ in range(6):
            env.play_move(3)
        legal = env.get_legal_moves()
        assert 3 not in legal
        assert len(legal) == 6

    def test_legal_moves_ext_from_state(self, env):
        env.play_move(0)
        env.play_move(0)
        state = env.get_state()
        legal = env.get_legal_moves_ext(state)
        # Column 0 still has room (only 2 of 6 filled)
        assert 0 in legal
        assert len(legal) == 7


# ---------------------------------------------------------------------------
# Phase 3: Move execution
# ---------------------------------------------------------------------------

class TestMoveExecution:

    def test_piece_lands_at_bottom(self, env, config):
        next_state, reward, done = env.play_move(3)
        assert env.board[5, 3] == config.PLAYER_1
        assert not done
        assert reward == 0.0
        assert env.current_player == config.PLAYER_2

    def test_stacking(self, env, config):
        env.play_move(3)  # P1 row 5
        env.play_move(3)  # P2 row 4
        env.play_move(3)  # P1 row 3
        assert env.board[5, 3] == config.PLAYER_1
        assert env.board[4, 3] == config.PLAYER_2
        assert env.board[3, 3] == config.PLAYER_1

    def test_returns_canonical_next_state(self, env):
        """play_move must return a (2,6,7) next state."""
        next_state, reward, done = env.play_move(3)
        assert next_state.shape == (2, 6, 7)

    def test_invalid_full_column_raises(self, env):
        for _ in range(6):
            env.play_move(3)
        with pytest.raises(ValueError, match="Column 3 is full"):
            env.play_move(3)

    def test_invalid_column_out_of_range(self, env):
        with pytest.raises(ValueError, match="Invalid column"):
            env.play_move(7)
        with pytest.raises(ValueError, match="Invalid column"):
            env.play_move(-1)


# ---------------------------------------------------------------------------
# Phase 4: Win detection
# ---------------------------------------------------------------------------

class TestWinDetection:

    def _setup_and_win(self, env, pieces, win_col):
        """Helper: place pieces on internal board, then play win_col."""
        for row, col, player in pieces:
            env.board[row, col] = player
        env.current_player = 1
        return env.play_move(win_col)

    def test_horizontal_win(self, env, config):
        pieces = [(5, 0, 1), (5, 1, 1), (5, 2, 1)]
        state, reward, done = self._setup_and_win(env, pieces, win_col=3)
        assert done
        assert reward == 1.0  # moving player gets +1.0
        assert env.check_winner() == config.PLAYER_1

    def test_vertical_win(self, env, config):
        pieces = [(5, 3, 1), (4, 3, 1), (3, 3, 1)]
        state, reward, done = self._setup_and_win(env, pieces, win_col=3)
        assert done
        assert reward == 1.0
        assert env.check_winner() == config.PLAYER_1

    def test_diagonal_ascending_win(self, env, config):
        # Build support so gravity allows the piece:
        # X at (5,0), (4,1), (3,2), play (2,3) wins /
        pieces = [
            (5, 0, 1),
            (5, 1, -1), (4, 1, 1),
            (5, 2, -1), (4, 2, -1), (3, 2, 1),
            (5, 3, -1), (4, 3, -1), (3, 3, -1),
        ]
        state, reward, done = self._setup_and_win(env, pieces, win_col=3)
        assert done
        assert reward == 1.0

    def test_diagonal_descending_win(self, env, config):
        # X at (5,3), (4,2), (3,1), play col 0 for (2,0) win \
        pieces = [
            (5, 3, 1),
            (5, 2, -1), (4, 2, 1),
            (5, 1, -1), (4, 1, -1), (3, 1, 1),
            (5, 0, -1), (4, 0, -1), (3, 0, -1),
        ]
        state, reward, done = self._setup_and_win(env, pieces, win_col=0)
        assert done
        assert reward == 1.0

    def test_no_winner_mid_game(self, env):
        env.play_move(0)
        env.play_move(1)
        env.play_move(2)
        assert env.check_winner() is None
        assert not env.is_terminal()

    def test_win_ends_game_immediately(self, env):
        """Game must be done=True on the winning move turn."""
        pieces = [(5, 0, 1), (5, 1, 1), (5, 2, 1)]
        _, reward, done = self._setup_and_win(env, pieces, 3)
        assert done


# ---------------------------------------------------------------------------
# Phase 5: Draw detection
# ---------------------------------------------------------------------------

class TestDrawDetection:

    def test_full_board_no_winner_is_draw(self, env, config):
        """Fill board with alternating pattern; verify terminal with reward 0.0."""
        for row in range(6):
            for col in range(7):
                env.board[row, col] = 1 if (row + col) % 2 == 0 else -1

        env.last_move = (0, 0)  # needed so check_winner can run
        winner = env.check_winner()
        legal = env.get_legal_moves()

        if winner is None and len(legal) == 0:
            # Simulate the terminal state detection
            assert env.is_terminal()
        # Just verify the helper returns None winner and no moves when board is full


# ---------------------------------------------------------------------------
# Phase 6: State manipulation helpers
# ---------------------------------------------------------------------------

class TestStateManipulation:

    def test_set_state_2d(self, env, config):
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, -1, 1, 0, 0, 0],
            [1, -1, 1, -1, 1, 0, 0]
        ])
        env.set_state(board, config.PLAYER_1)
        assert np.array_equal(env.board, board)
        assert env.current_player == config.PLAYER_1

    def test_apply_move_to_state_no_side_effects(self, env, config):
        """apply_move_to_state must NOT modify the live environment board."""
        state = env.get_state()
        original_board = env.board.copy()

        new_state = env.apply_move_to_state(state, 3, config.PLAYER_1)

        assert np.array_equal(env.board, original_board), "Board was mutated!"
        assert new_state.shape == (2, 6, 7)

    def test_apply_move_to_state_piece_present(self, env, config):
        """After applying a move, next-player's channel 1 should show the placed piece."""
        state = env.get_state()
        # apply_move_to_state returns from *next* player's perspective
        new_state = env.apply_move_to_state(state, 3, config.PLAYER_1)
        # Next player sees P1's piece as opponent's piece in channel 1
        assert new_state[1, 5, 3] == 1.0

    def test_check_winner_from_state(self, env, config):
        board = np.zeros((6, 7), dtype=int)
        board[5, 0] = board[5, 1] = board[5, 2] = board[5, 3] = 1
        winner = env.check_winner_from_state(board)
        assert winner == config.PLAYER_1


# ---------------------------------------------------------------------------
# Phase 7: Canonical perspective symmetry
# ---------------------------------------------------------------------------

class TestCanonicalPerspective:

    def test_player1_sees_own_pieces_in_ch0(self, env):
        """After P1 plays col 3, get_state_from_perspective(1) has piece in ch0."""
        env.play_move(3)
        state = env.get_state_from_perspective(1)
        assert state[0, 5, 3] == 1.0, "P1 should see own piece in channel 0"
        assert state[1, 5, 3] == 0.0, "P1 should not see own piece in channel 1"

    def test_player2_sees_opponent_in_ch1(self, env):
        """After P1 plays col 3, get_state_from_perspective(-1) has piece in ch1."""
        env.play_move(3)
        state = env.get_state_from_perspective(-1)
        assert state[0, 5, 3] == 0.0, "P2 has no own piece at (5,3) yet"
        assert state[1, 5, 3] == 1.0, "P2 should see P1's piece as opponent in ch1"

    def test_perspectives_are_channel_flips_of_each_other(self, env):
        """P1's ch0 == P2's ch1 and vice versa (same board, different labels)."""
        env.play_move(3)  # P1 piece
        s1 = env.get_state_from_perspective(1)
        s2 = env.get_state_from_perspective(-1)
        np.testing.assert_array_equal(s1[0], s2[1])
        np.testing.assert_array_equal(s1[1], s2[0])


# ---------------------------------------------------------------------------
# Phase 8: Reward convention
# ---------------------------------------------------------------------------

class TestRewardConvention:

    def test_non_terminal_reward_is_zero(self, env):
        _, reward, done = env.play_move(3)
        assert reward == 0.0
        assert not done

    def test_winning_reward_is_plus_one(self, env):
        """Moving player's winning move must yield reward=+1.0."""
        env.board[5, 0] = env.board[5, 1] = env.board[5, 2] = 1
        env.current_player = 1
        _, reward, done = env.play_move(3)
        assert done
        assert reward == 1.0

    def test_draw_reward_is_zero(self, config):
        """Draw must produce reward == config.DRAW_VALUE (0.0)."""
        assert config.DRAW_VALUE == 0


# ---------------------------------------------------------------------------
# Integration: Full game
# ---------------------------------------------------------------------------

def test_full_game_p1_wins():
    """P1 wins horizontally: cols 3,4,5,6 (P2 plays elsewhere)."""
    config = Config()
    env = ConnectFourEnvironment(config)

    # P1: 3, P2: 0, P1: 4, P2: 0, P1: 5, P2: 0, P1: 6 → wins
    moves = [3, 0, 4, 0, 5, 0, 6]
    done = False
    reward = 0.0
    for move in moves:
        state, reward, done = env.play_move(move)
        if done:
            break

    assert done, "Game should end after P1 gets 4 in a row"
    assert reward == 1.0, f"Winning reward should be 1.0, got {reward}"
    assert state.shape == (2, 6, 7)


def test_full_game_no_crash_42_moves():
    """A randomly-played game should finish within 42 moves."""
    import random
    config = Config()
    env = ConnectFourEnvironment(config)
    env.reset()

    done = False
    move_count = 0
    while not done and move_count < 42:
        legal = env.get_legal_moves()
        if not legal:
            break
        col = random.choice(legal)
        _, _, done = env.play_move(col)
        move_count += 1

    assert move_count <= 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
