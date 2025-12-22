"""
DQN Value Network - Neural Network for Q-Value Estimation

This module implements the Q-network used in DQN (Deep Q-Network) algorithm.
The network takes a state as input and outputs estimated Q-values for ALL actions.

Architecture:
-------------
Input: State (2, 6, 7) - 2 channels (canonical representation)
   - Channel 0: Current player's pieces (MY pieces)
   - Channel 1: Opponent's pieces (THEIR pieces)
   ↓
Convolutional Layers (extract spatial features)
   ↓
Flatten
   ↓
Fully Connected Layers (configurable)
   ↓
Output: Q-values [Q(s,0), Q(s,1), ..., Q(s,6)] - one per action

Key Design Decisions:
--------------------
1. **Multi-head output**: One Q-value per action (efficient - single forward pass)
2. **Configurable architecture**: Adjustable layers, dimensions, dropout
3. **Spatial features**: Convolutional layers for board patterns
4. **Regularization**: Dropout + L2 weight decay support

Example:
--------
    network = DQNValueNetwork(
        input_channels=2,
        num_actions=7,
        conv_channels=[32, 64],
        fc_dims=[128, 64],
        dropout_rate=0.1
    )
    
    state = torch.tensor(state, dtype=torch.float32)  # (batch, 2, 6, 7)
    q_values = network(state)  # (batch, 7) - Q-value for each action
    
    best_action = q_values.argmax(dim=1)  # Choose action with highest Q-value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class DQNValueNetwork(nn.Module):
    """
    Deep Q-Network for estimating action values Q(s,a).
    
    This network uses convolutional layers to extract spatial features from
    the board state, then fully connected layers to estimate Q-values for
    each possible action.
    
    Architecture:
        Conv layers → Flatten → FC layers → Output (7 Q-values)
    
    Attributes:
        input_channels (int): Number of input channels (2 for canonical Connect 4)
        num_actions (int): Number of possible actions (7 for Connect 4)
        conv_channels (List[int]): Channels for each conv layer
        fc_dims (List[int]): Dimensions for each FC layer
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self,
                 input_channels: int = 2,  # I think this is the num_players
                 num_actions: int = 7,     # Cols in board
                 conv_channels: Optional[List[int]] = None,
                 fc_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.0,
                 board_height: int = 6,
                 board_width: int = 7):
        """
        Initialize DQN Value Network.
        
        Args:
            input_channels: Number of input channels (2 for canonical Connect 4)
            num_actions: Number of possible actions (7 columns)
            conv_channels: List of channel sizes for conv layers.
                          Default: [32, 64] (2 conv layers)
                          Example: [32, 64, 128] for 3 conv layers
            fc_dims: List of dimensions for fully connected layers.
                    Default: [128] (1 FC layer before output)
                    Example: [256, 128] for 2 FC layers
            dropout_rate: Dropout probability (0.0 = no dropout, 0.5 = 50% dropout)
                         Applied after each FC layer except the last
            board_height: Height of game board (6 for Connect 4)
            board_width: Width of game board (7 for Connect 4)
        
        Example:
            # Simple network
            network = DQNValueNetwork()
            
            # Deeper network with more capacity
            network = DQNValueNetwork(
                conv_channels=[32, 64, 128],
                fc_dims=[256, 128],
                dropout_rate=0.2
            )
            
            # Shallow network (faster, less capacity)
            network = DQNValueNetwork(
                conv_channels=[32],
                fc_dims=[64],
                dropout_rate=0.0
            )
        """
        super(DQNValueNetwork, self).__init__()
        
        # Store configuration
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.board_height = board_height
        self.board_width = board_width
        self.dropout_rate = dropout_rate
        
        # Default architectures if not specified
        if conv_channels is None:
            conv_channels = [32, 64]  # 2 conv layers
        if fc_dims is None:
            fc_dims = [128]  # 1 FC layer
        
        self.conv_channels = conv_channels
        self.fc_dims = fc_dims
        
        # Build convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Calculate size after conv layers
        conv_output_size = self._calculate_conv_output_size()
        
        # Build fully connected layers
        self.fc_layers = self._build_fc_layers(conv_output_size)
        
        # Output layer: one Q-value per action
        last_fc_dim = fc_dims[-1] if fc_dims else conv_output_size
        self.output_layer = nn.Linear(last_fc_dim, num_actions)
        
        # Initialize weights
        self._initialize_weights()
    
    def _build_conv_layers(self) -> nn.ModuleList:
        """
        Build convolutional layers for spatial feature extraction.
        
        Each conv layer:
        - 3x3 kernel with padding=1 (preserves spatial dimensions)
        - BatchNorm for training stability
        - ReLU activation
        
        Returns:
            nn.ModuleList: List of conv layer modules
        """
        layers = nn.ModuleList()
        
        in_channels = self.input_channels
        for out_channels in self.conv_channels:
            # Conv layer: 3x3 kernel, padding=1 preserves size
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            bn = nn.BatchNorm2d(out_channels)
            
            layers.append(nn.ModuleDict({
                'conv': conv,
                'bn': bn
            }))
            
            in_channels = out_channels
        
        return layers
    
    def _calculate_conv_output_size(self) -> int:
        """
        Calculate the flattened size after conv layers.
        
        Since we use padding=1 with kernel=3, spatial dimensions are preserved.
        Output size = last_conv_channels * height * width
        
        Returns:
            int: Flattened size after conv layers
        """
        if self.conv_channels:
            last_conv_channels = self.conv_channels[-1]
        else:
            last_conv_channels = self.input_channels
        
        return last_conv_channels * self.board_height * self.board_width
    
    def _build_fc_layers(self, input_size: int) -> nn.ModuleList:
        """
        Build fully connected layers.
        
        Each FC layer (except last):
        - Linear transformation
        - ReLU activation
        - Optional dropout for regularization
        
        Args:
            input_size: Size of flattened conv output
        
        Returns:
            nn.ModuleList: List of FC layer modules
        """
        layers = nn.ModuleList()
        
        in_features = input_size
        for out_features in self.fc_dims:
            fc = nn.Linear(in_features, out_features)
            
            layers.append(nn.ModuleDict({
                'fc': fc,
                'dropout': nn.Dropout(self.dropout_rate) if self.dropout_rate > 0 else None
            }))
            
            in_features = out_features
        
        return layers
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        He initialization is recommended for ReLU networks:
        - Conv layers: He normal
        - Linear layers: He normal
        - Biases: zeros
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, 2, 6, 7)
               - Channel 0: Current player's pieces (MY pieces)
               - Channel 1: Opponent's pieces (THEIR pieces)
        
        Returns:
            torch.Tensor: Q-values for each action, shape (batch_size, 7)
                         Each value represents Q(s, a) for that action
        
        Example:
            state = torch.randn(32, 2, 6, 7)  # Batch of 32 states
            q_values = network(state)          # (32, 7)
            
            # Get best action for each state
            best_actions = q_values.argmax(dim=1)  # (32,)
            
            # Get Q-value for specific actions
            actions = torch.tensor([3, 1, 4, ...])  # (32,)
            q_selected = q_values.gather(1, actions.unsqueeze(1))  # (32, 1)
        """
        # Convolutional layers
        for layer_dict in self.conv_layers:
            x = layer_dict['conv'](x)
            x = layer_dict['bn'](x)
            x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for layer_dict in self.fc_layers:
            x = layer_dict['fc'](x)
            x = F.relu(x)
            if layer_dict['dropout'] is not None:
                x = layer_dict['dropout'](x)
        
        # Output layer (no activation - raw Q-values)
        q_values = self.output_layer(x)
        
        return q_values
    
    def get_config(self) -> dict:
        """
        Get network configuration.
        
        Returns:
            dict: Configuration parameters
        
        Example:
            config = network.get_config()
            print(f"Architecture: {config}")
        """
        return {
            'input_channels': self.input_channels,
            'num_actions': self.num_actions,
            'conv_channels': self.conv_channels,
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            'board_height': self.board_height,
            'board_width': self.board_width,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def __repr__(self) -> str:
        """String representation of network."""
        config = self.get_config()
        return (f"DQNValueNetwork(\n"
                f"  conv_channels={config['conv_channels']},\n"
                f"  fc_dims={config['fc_dims']},\n"
                f"  dropout={config['dropout_rate']},\n"
                f"  parameters={config['total_parameters']:,}\n"
                f")")