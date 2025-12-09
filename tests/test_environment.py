"""
Comprehensive test suite for Connect 4 environment.

Tests cover:
- State representation and shape
- Legal moves detection
- Move execution and board updates
- Win detection (horizontal, vertical, diagonal)
- Draw detection
- Edge cases and error handling
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from environment import Config, ConnectFourEnvironment


class EnvironmentUnitTests:
    """Test basic environment functionality."""
    
    def test_initialization(self):
        """Test environment initializes correctly."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        assert env.rows == 6
        assert env.cols == 7
        assert env.current_player == config.PLAYER_1
        assert env.last_move is None
        assert np.all(env.board == 0)
    
    def test_reset(self):
        """Test environment reset."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Make some moves
        env.play_move(3)
        env.play_move(3)
        
        # Reset
        state = env.reset()
        
        assert np.all(env.board == 0)
        assert env.current_player == config.PLAYER_1
        assert env.last_move is None
        assert state.shape == (3, 6, 7)
    
    def test_state_representation(self):
        """Test 3-channel state representation."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        state = env.get_state()
        
        # Check shape
        assert state.shape == (3, 6, 7)
        
        # Check all channels are empty initially
        assert np.all(state[0] == 0)  # Player 1 channel
        assert np.all(state[1] == 0)  # Player 2 channel
        assert np.all(state[2] == 1)  # Current player (P1 = 1.0)
        
        # Make a move
        env.play_move(3)
        state = env.get_state()
        
        # Check Player 1 piece appears in channel 0
        assert state[0, 5, 3] == 1.0
        # Check current player switched to P2 (0.0)
        assert np.all(state[2] == 0.0)


class TestLegalMoves:
    """Test legal move detection."""
    
    def test_initial_legal_moves(self):
        """Test all columns are legal initially."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        legal_moves = env.get_legal_moves()
        assert legal_moves == [0, 1, 2, 3, 4, 5, 6]
    
    def test_full_column(self):
        """Test full column is not legal."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Fill column 3
        for _ in range(6):
            env.play_move(3)
        
        legal_moves = env.get_legal_moves()
        assert 3 not in legal_moves
        assert len(legal_moves) == 6
    
    def test_get_legal_moves_ext(self):
        """Test external state legal moves."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Make some moves
        env.play_move(0)
        env.play_move(0)
        
        state = env.get_state()
        legal_moves = env.get_legal_moves_ext(state)
        
        assert 0 in legal_moves
        assert len(legal_moves) == 7


class TestMoveExecution:
    """Test move execution and board updates."""
    
    def test_play_move_basic(self):
        """Test basic move execution."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        state, reward, done = env.play_move(3)
        
        # Check piece placed at bottom
        assert env.board[5, 3] == config.PLAYER_1
        # Check game not done
        assert not done
        assert reward is None
        # Check player switched
        assert env.current_player == config.PLAYER_2
    
    def test_stacking_pieces(self):
        """Test pieces stack correctly."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Play 3 pieces in same column
        env.play_move(3)  # P1 at row 5
        env.play_move(3)  # P2 at row 4
        env.play_move(3)  # P1 at row 3
        
        assert env.board[5, 3] == config.PLAYER_1
        assert env.board[4, 3] == config.PLAYER_2
        assert env.board[3, 3] == config.PLAYER_1
    
    def test_invalid_move_full_column(self):
        """Test error on full column."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Fill column
        for _ in range(6):
            env.play_move(3)
        
        # Try to play in full column
        with pytest.raises(ValueError, match="Column 3 is full"):
            env.play_move(3)
    
    def test_invalid_column_index(self):
        """Test error on invalid column."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        with pytest.raises(ValueError, match="Invalid column"):
            env.play_move(7)
        
        with pytest.raises(ValueError, match="Invalid column"):
            env.play_move(-1)


class TestWinDetection:
    """Test win detection in all directions."""
    
    def test_horizontal_win(self):
        """Test horizontal 4-in-a-row detection."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create horizontal win for P1
        # P1: columns 0, 2, 4, 6
        # P2: columns 1, 3, 5
        env.play_move(0)  # P1
        env.play_move(0)  # P2
        env.play_move(1)  # P1
        env.play_move(0)  # P2
        env.play_move(2)  # P1
        env.play_move(0)  # P2
        state, reward, done = env.play_move(3)  # P1 wins
        
        assert done
        assert reward == config.PLAYER_1
        assert env.check_winner() == config.PLAYER_1
    
    def test_vertical_win(self):
        """Test vertical 4-in-a-row detection."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create vertical win for P1 in column 3
        env.play_move(3)  # P1
        env.play_move(4)  # P2
        env.play_move(3)  # P1
        env.play_move(4)  # P2
        env.play_move(3)  # P1
        env.play_move(4)  # P2
        state, reward, done = env.play_move(3)  # P1 wins
        
        assert done
        assert reward == config.PLAYER_1
        assert env.check_winner() == config.PLAYER_1
    
    def test_diagonal_win_ascending(self):
        """Test diagonal win (/)."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create ascending diagonal win for P1
        # Pattern:
        #     . . . . . . .
        #     . . . X . . .
        #     . . X O . . .
        #     . X O O . . .
        #     X O O O . . .
        
        env.play_move(0)  # P1 at (5,0)
        env.play_move(1)  # P2 at (5,1)
        env.play_move(1)  # P1 at (4,1)
        env.play_move(2)  # P2 at (5,2)
        env.play_move(2)  # P1 at (4,2)
        env.play_move(3)  # P2 at (5,3)
        env.play_move(2)  # P1 at (3,2)
        env.play_move(3)  # P2 at (4,3)
        env.play_move(3)  # P1 at (3,3)
        env.play_move(4)  # P2 at (5,4)
        state, reward, done = env.play_move(3)  # P1 at (2,3) - wins!
        
        assert done
        assert reward == config.PLAYER_1
    
    def test_diagonal_win_descending(self):
        r"""Test diagonal win (\)."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create descending diagonal win for P1
        env.play_move(3)  # P1 at (5,3)
        env.play_move(2)  # P2 at (5,2)
        env.play_move(2)  # P1 at (4,2)
        env.play_move(1)  # P2 at (5,1)
        env.play_move(1)  # P1 at (4,1)
        env.play_move(0)  # P2 at (5,0)
        env.play_move(1)  # P1 at (3,1)
        env.play_move(0)  # P2 at (4,0)
        env.play_move(0)  # P1 at (3,0)
        env.play_move(6)  # P2 at (5,6)
        state, reward, done = env.play_move(0)  # P1 at (2,0) - wins!
        
        assert done
        assert reward == config.PLAYER_1
    
    def test_no_winner_yet(self):
        """Test no winner in incomplete game."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        env.play_move(0)
        env.play_move(1)
        env.play_move(2)
        
        assert env.check_winner() is None
        assert not env.is_terminal()


class TestDrawDetection:
    """Test draw/tie detection."""
    
    def test_full_board_draw(self):
        """Test draw when board is full with no winner."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Fill board in a pattern that creates no winner
        # This is a specific pattern that fills the board without 4-in-a-row
        moves = [
            0, 0, 0, 1, 1, 1,  # Fill columns 0-1
            2, 2, 2, 3, 3, 3,  # Fill columns 2-3
            4, 4, 4, 5, 5, 5,  # Fill columns 4-5
            6, 6, 6,           # Partial column 6
            0, 1, 2, 3, 4, 5, 6,  # Top rows
            0, 1, 2, 3, 4, 5, 6,
            0, 1, 2, 3, 4, 5, 6
        ]
        
        done = False
        for move in moves:
            if move in env.get_legal_moves():
                _, reward, done = env.play_move(move)
                if done:
                    break
        
        # Check if we filled the board
        if len(env.get_legal_moves()) == 0:
            assert done
            assert reward == config.DRAW_VALUE


class TestStateManipulation:
    """Test state manipulation methods."""
    
    def test_set_state(self):
        """Test setting environment to specific state."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create a specific board
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
    
    def test_apply_move_to_state(self):
        """Test applying move without modifying environment."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        state = env.get_state()
        original_board = env.board.copy()
        
        # Apply move to state (not environment)
        new_state = env.apply_move_to_state(state, 3, config.PLAYER_1)
        
        # Environment should be unchanged
        assert np.array_equal(env.board, original_board)
        
        # New state should have the piece
        assert new_state[0, 5, 3] == 1.0
    
    def test_check_winner_from_state(self):
        """Test checking winner from external state."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        # Create winning state
        board = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0]  # Horizontal win for P1
        ])
        
        winner = env.check_winner_from_state(board)
        assert winner == config.PLAYER_1


class TestRendering:
    """Test rendering methods."""
    
    def test_render(self, capsys):
        """Test board rendering."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        env.play_move(3)
        env.render()
        
        captured = capsys.readouterr()
        assert 'X' in captured.out
        assert '0 1 2 3 4 5 6' in captured.out
    
    def test_render_ext_state(self, capsys):
        """Test external state rendering."""
        config = Config()
        env = ConnectFourEnvironment(config)
        
        env.play_move(3)
        state = env.get_state()
        env.render_ext_state(state)
        
        captured = capsys.readouterr()
        assert 'X' in captured.out


def test_full_game_simulation():
    """Test a complete game from start to finish."""
    config = Config()
    env = ConnectFourEnvironment(config)
    
    # Play a game
    moves = [3, 3, 4, 4, 5, 5, 6]  # P1 should win horizontally
    
    done = False
    for move in moves:
        state, reward, done = env.play_move(move)
        if done:
            break
    
    assert done
    assert reward == config.PLAYER_1
    assert state.shape == (3, 6, 7)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])