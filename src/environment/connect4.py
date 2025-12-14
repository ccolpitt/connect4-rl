"""
Connect 4 game environment for reinforcement learning.

This module implements the Connect 4 game logic with a state representation
optimized for neural network training. The environment uses a 3-channel
representation: Player 1 pieces, Player 2 pieces, and current player indicator.
"""

import numpy as np
from typing import Tuple, List, Optional


class ConnectFourEnvironment:
    """
    Connect 4 game environment with NN-friendly state representation.
    
    State Representation:
        3D tensor of shape (3, ROWS, COLS) where:
        - Channel 0: Binary mask of Player 1 pieces (1.0 where present, 0.0 elsewhere)
        - Channel 1: Binary mask of Player 2 pieces (1.0 where present, 0.0 elsewhere)
        - Channel 2: Current player indicator (1.0 for Player 1, 0.0 for Player 2)
    
    Board Representation (internal):
        2D array of shape (ROWS, COLS) where:
        - 0 = empty cell
        - 1 = Player 1 piece
        - -1 = Player 2 piece
    
    Attributes:
        config: Configuration object with game parameters
        rows (int): Number of rows (6)
        cols (int): Number of columns (7)
        board (np.ndarray): Internal 2D board state
        current_player (int): Current player (1 or -1)
        last_move (tuple): Last move coordinates (row, col) or None
    """
    
    def __init__(self, config):
        """
        Initialize Connect 4 environment.
        
        Args:
            config: Configuration object with ROWS, COLS, PLAYER_1, PLAYER_2, DRAW_VALUE
        """
        self.config = config
        self.rows = config.ROWS
        self.cols = config.COLS
        self.board = np.zeros((config.ROWS, config.COLS), dtype=int)
        self.current_player = config.PLAYER_1
        self.last_move = None
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial state representation (3, ROWS, COLS)
        """
        self.board.fill(0)
        self.current_player = self.config.PLAYER_1
        self.last_move = None
        return self.get_state()
    
    def get_state(self) -> np.ndarray:
        """
        Get current state from current player's perspective (canonical representation).
        
        CRITICAL: For self-play, the agent must always see the board from its own
        perspective, with its pieces in Channel 0 and opponent's in Channel 1.
        
        This is the "canonical" representation where the network always sees:
        - Channel 0: MY pieces (current player)
        - Channel 1: OPPONENT's pieces
        
        Returns:
            np.ndarray: State tensor of shape (2, ROWS, COLS) in (C, H, W) format
                - Channel 0: Current player's pieces (MY pieces)
                - Channel 1: Opponent's pieces (THEIR pieces)
        """
        return self.get_state_from_perspective(self.current_player)
    
    def get_state_from_perspective(self, player: int) -> np.ndarray:
        """
        Get state from specified player's perspective (canonical representation).
        
        This ensures the agent always sees:
        - Channel 0: Its own pieces
        - Channel 1: Opponent's pieces
        
        No player indicator channel is needed because the network always sees
        the board from "my" perspective (canonical representation).
        
        Args:
            player: Player to get perspective for (1 or -1)
        
        Returns:
            np.ndarray: State from player's perspective (2, ROWS, COLS)
        """
        state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        
        if player == self.config.PLAYER_1:
            # Player 1's perspective
            state[0, :, :] = (self.board == self.config.PLAYER_1).astype(np.float32)  # My pieces
            state[1, :, :] = (self.board == self.config.PLAYER_2).astype(np.float32)  # Opponent's pieces
        else:
            # Player 2's perspective (flipped)
            state[0, :, :] = (self.board == self.config.PLAYER_2).astype(np.float32)  # My pieces
            state[1, :, :] = (self.board == self.config.PLAYER_1).astype(np.float32)  # Opponent's pieces
        
        return state
    
    def get_current_player(self) -> int:
        """
        Get the current player.
        
        Returns:
            int: Current player (1 or -1)
        """
        return self.current_player
    
    def get_legal_moves(self) -> List[int]:
        """
        Get list of legal column indices where pieces can be placed.
        
        Returns:
            List[int]: List of valid column indices (0 to COLS-1)
        """
        return [col for col in range(self.cols) if self.board[0, col] == 0]
    
    def get_legal_moves_ext(self, state: np.ndarray) -> List[int]:
        """
        Get legal moves from external state representation.
        
        Args:
            state: State tensor of shape (2, ROWS, COLS) or (ROWS, COLS)
        
        Returns:
            List[int]: List of valid column indices
        """
        if state.shape == (2, self.rows, self.cols):
            # Extract board from 2-channel state
            board = np.zeros((self.rows, self.cols))
            board[state[0, :, :] == 1] = self.config.PLAYER_1
            board[state[1, :, :] == 1] = self.config.PLAYER_2
        else:
            # Fallback for 2D board
            board = state
        
        return [col for col in range(self.cols) if board[0, col] == 0]
    
    def play_move(self, col: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute a move in the specified column.
        
        Args:
            col: Column index (0 to COLS-1)
        
        Returns:
            Tuple containing:
                - state (np.ndarray): New state from NEXT player's perspective
                - reward (float): Reward from PREVIOUS player's perspective
                  (+1.0 for win, -1.0 for loss, 0.0 for draw or continuing game)
                - done (bool): Whether the game has ended
        
        Raises:
            ValueError: If the column is full or invalid
        """
        if col < 0 or col >= self.cols:
            raise ValueError(f"Invalid column {col}. Must be between 0 and {self.cols-1}")
        
        if self.board[0, col] != 0:
            raise ValueError(f"Column {col} is full!")
        
        # Store who made this move (before switching players)
        moving_player = self.current_player
        
        # Make the move (drop piece to lowest available row)
        for row in reversed(range(self.rows)):
            if self.board[row, col] == 0:
                self.board[row, col] = self.current_player
                self.last_move = (row, col)
                break
        
        # Check game state
        winner = self.check_winner()
        legal_moves = self.get_legal_moves()
        done = winner is not None or len(legal_moves) == 0
        
        # Determine reward from perspective of player who just moved
        if winner is not None:
            # If there's a winner, it must be the player who just moved
            # Reward is +1.0 for winning, -1.0 for losing
            reward = +1.0 if winner == moving_player else -1.0
        elif len(legal_moves) == 0:
            reward = self.config.DRAW_VALUE  # Draw (0.0)
        else:
            reward = 0.0  # Game continues, no immediate reward
        
        # Switch to next player
        self.current_player *= -1
        
        # Return state from NEXT player's perspective
        return self.get_state(), reward, done
    
    def apply_move_to_state(self, state: np.ndarray, action: int,
                           current_player: int) -> np.ndarray:
        """
        Apply move to state without modifying environment (for MCTS simulation).
        
        Args:
            state: Current state tensor (2, ROWS, COLS)
            action: Column index to play
            current_player: Player making the move (1 or -1)
        
        Returns:
            np.ndarray: New state after applying the move (from next player's perspective)
        """
        # Extract board from state
        board = np.zeros((self.rows, self.cols))
        board[state[0, :, :] == 1] = current_player  # Channel 0 = current player
        board[state[1, :, :] == 1] = -current_player  # Channel 1 = opponent
        
        # Apply move
        for row in reversed(range(self.rows)):
            if board[row, action] == 0:
                board[row, action] = current_player
                break
        
        # Return state from NEXT player's perspective (flipped)
        next_player = -current_player
        new_state = np.zeros((2, self.rows, self.cols), dtype=np.float32)
        new_state[0, :, :] = (board == next_player).astype(np.float32)  # Next player's pieces
        new_state[1, :, :] = (board == current_player).astype(np.float32)  # Current player's pieces
        
        return new_state
    
    def is_terminal(self) -> bool:
        """
        Check if current state is terminal (game over).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.check_winner() is not None or not self.get_legal_moves()
    
    def check_winner(self) -> Optional[int]:
        """
        Check if there is a winner based on the last move.
        
        This method efficiently checks only around the last played piece
        rather than scanning the entire board.
        
        Returns:
            int or None: Winner player (1 or -1), or None if no winner yet
        """
        if self.last_move is None:
            return None
        
        row, col = self.last_move
        player = self.board[row, col]
        
        # Check all four directions: vertical, horizontal, diagonal /, diagonal \
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            # Check both directions along this line
            for direction in [1, -1]:
                r, c = row + dr * direction, col + dc * direction
                while (0 <= r < self.rows and 0 <= c < self.cols and 
                       self.board[r, c] == player):
                    count += 1
                    r += dr * direction
                    c += dc * direction
            
            if count >= 4:
                return player
        
        return None
    
    def check_winner_from_state(self, state: np.ndarray) -> Optional[int]:
        """
        Check winner from external state without modifying environment.
        
        This method scans the entire board since we don't have last_move info.
        
        Args:
            state: State tensor (2, ROWS, COLS) or 2D board
        
        Returns:
            int or None: Winner player (1 or -1), or None if no winner
        """
        # Extract 2D board from 2-channel state
        if len(state.shape) == 3 and state.shape == (2, self.rows, self.cols):
            board = np.zeros((self.rows, self.cols))
            board[state[0, :, :] == 1] = self.config.PLAYER_1
            board[state[1, :, :] == 1] = self.config.PLAYER_2
        else:
            board = state
        
        # Check entire board for wins
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row, col] != 0:
                    player = board[row, col]
                    
                    for dr, dc in directions:
                        count = 1
                        # Check positive direction
                        r, c = row + dr, col + dc
                        while (0 <= r < self.rows and 0 <= c < self.cols and 
                               board[r, c] == player):
                            count += 1
                            r, c = r + dr, c + dc
                        
                        # Check negative direction
                        r, c = row - dr, col - dc
                        while (0 <= r < self.rows and 0 <= c < self.cols and 
                               board[r, c] == player):
                            count += 1
                            r, c = r - dr, c - dc
                        
                        if count >= 4:
                            return player
        
        return None
    
    def set_state(self, board: np.ndarray, current_player: int) -> None:
        """
        Set environment to a specific board state.
        
        Args:
            board: 2D board array or 2-channel state tensor
            current_player: Player whose turn it is (1 or -1)
        """
        if len(board.shape) == 3 and board.shape == (2, self.rows, self.cols):
            # Extract board from 2-channel state
            self.board = np.zeros((self.rows, self.cols))
            self.board[board[0, :, :] == 1] = self.config.PLAYER_1
            self.board[board[1, :, :] == 1] = self.config.PLAYER_2
        else:
            # 2D board format
            self.board = board.copy()
        
        self.current_player = current_player
        self.last_move = None
    
    def render(self) -> None:
        """
        Print the current board state to console.
        
        Display format:
            X = Player 1
            O = Player 2
            . = Empty
        """
        display = {1: 'X', -1: 'O', 0: '.'}
        print()
        for row in self.board:
            print(' '.join(display[val] for val in row))
        print('0 1 2 3 4 5 6')
        print()
    
    def render_ext_state(self, state: np.ndarray) -> None:
        """
        Render external state representation.
        
        Args:
            state: State tensor (2, ROWS, COLS) or 2D board
        """
        display = {1: 'X', -1: 'O', 0: '.'}
        
        # Extract board from state
        if len(state.shape) == 3 and state.shape == (2, self.rows, self.cols):
            board = np.zeros((self.rows, self.cols))
            board[state[0, :, :] == 1] = self.config.PLAYER_1
            board[state[1, :, :] == 1] = self.config.PLAYER_2
        elif len(state.shape) == 2 and state.shape == (self.rows, self.cols):
            board = state
        else:
            # Flat format
            board_flat = state[0:self.rows * self.cols]
            board = board_flat.reshape((self.rows, self.cols))
        
        print()
        for row in board:
            print(' '.join(display[int(val)] for val in row))
        print('0 1 2 3 4 5 6')
        print()