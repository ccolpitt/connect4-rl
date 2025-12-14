"""
Actor-Critic Network - Shared Backbone with Dual Heads

This module implements a neural network for A2C (Advantage Actor-Critic) algorithm.
The network uses a shared convolutional backbone that branches into two heads:
- Actor head: Outputs action probabilities (policy)
- Critic head: Outputs state value estimate

Architecture:
-------------
Input: State (3, 6, 7) - 3 channels (Player1, Player2, CurrentPlayer)
   ↓
Shared Convolutional Layers (extract spatial features)
   ↓
Shared Flatten
   ↓
Shared Fully Connected Layers
   ↓
   ╱────────╲
  ↓          ↓
Actor Head   Critic Head
(7 probs)    (1 value)

Key Design Decisions:
--------------------
1. **Shared backbone**: Both actor and critic learn same board features (efficient)
2. **Actor output**: Softmax probabilities over actions (stochastic policy)
3. **Critic output**: Single scalar value V(s) (state value estimate)
4. **Configurable architecture**: Adjustable layers, dimensions, dropout

Example:
--------
    network = ActorCriticNetwork(
        input_channels=3,
        num_actions=7,
        conv_channels=[32, 64],
        fc_dims=[128],
        dropout_rate=0.1
    )
    
    state = torch.tensor(state, dtype=torch.float32)  # (batch, 3, 6, 7)
    action_probs, state_value = network(state)
    
    # Sample action from policy
    action = torch.multinomial(action_probs, 1)
    
    # Use value for advantage computation
    advantage = returns - state_value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network with shared backbone for A2C algorithm.
    
    This network uses convolutional layers to extract spatial features from
    the board state, then branches into two heads:
    - Actor: Outputs probability distribution over actions
    - Critic: Outputs state value estimate V(s)
    
    Architecture:
        Shared: Conv layers → Flatten → FC layers
        Actor: FC → Softmax (7 action probabilities)
        Critic: FC → Linear (1 state value)
    
    Attributes:
        input_channels (int): Number of input channels (3 for Connect 4)
        num_actions (int): Number of possible actions (7 for Connect 4)
        conv_channels (List[int]): Channels for each conv layer
        fc_dims (List[int]): Dimensions for shared FC layers
        dropout_rate (float): Dropout probability for regularization
    """
    
    def __init__(self,
                 input_channels: int = 3,
                 num_actions: int = 7,
                 conv_channels: Optional[List[int]] = None,
                 fc_dims: Optional[List[int]] = None,
                 dropout_rate: float = 0.0,
                 board_height: int = 6,
                 board_width: int = 7):
        """
        Initialize Actor-Critic Network.
        
        Args:
            input_channels: Number of input channels (3 for Connect 4)
            num_actions: Number of possible actions (7 columns)
            conv_channels: List of channel sizes for conv layers.
                          Default: [32, 64] (2 conv layers)
                          Example: [32, 64, 128] for 3 conv layers
            fc_dims: List of dimensions for shared fully connected layers.
                    Default: [128] (1 shared FC layer)
                    Example: [256, 128] for 2 shared FC layers
            dropout_rate: Dropout probability (0.0 = no dropout, 0.5 = 50% dropout)
                         Applied after each shared FC layer
            board_height: Height of game board (6 for Connect 4)
            board_width: Width of game board (7 for Connect 4)
        
        Example:
            # Simple network
            network = ActorCriticNetwork()
            
            # Deeper network with more capacity
            network = ActorCriticNetwork(
                conv_channels=[32, 64, 128],
                fc_dims=[256, 128],
                dropout_rate=0.2
            )
            
            # Shallow network (faster, less capacity)
            network = ActorCriticNetwork(
                conv_channels=[32],
                fc_dims=[64],
                dropout_rate=0.0
            )
        """
        super(ActorCriticNetwork, self).__init__()
        
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
            fc_dims = [128]  # 1 shared FC layer
        
        self.conv_channels = conv_channels
        self.fc_dims = fc_dims
        
        # Build shared convolutional layers
        self.conv_layers = self._build_conv_layers()
        
        # Calculate size after conv layers
        conv_output_size = self._calculate_conv_output_size()
        
        # Build shared fully connected layers
        self.shared_fc_layers = self._build_fc_layers(conv_output_size)
        
        # Get final shared layer dimension
        last_shared_dim = fc_dims[-1] if fc_dims else conv_output_size
        
        # Actor head: outputs action logits (before softmax)
        self.actor_head = nn.Linear(last_shared_dim, num_actions)
        
        # Critic head: outputs state value V(s)
        self.critic_head = nn.Linear(last_shared_dim, 1)
        
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
        Build shared fully connected layers.
        
        Each FC layer:
        - Linear transformation
        - ReLU activation
        - Optional dropout for regularization
        
        Args:
            input_size: Size of flattened conv output
        
        Returns:
            nn.ModuleList: List of shared FC layer modules
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
        Initialize network weights using orthogonal initialization.
        
        Orthogonal initialization is recommended for actor-critic networks:
        - Conv layers: Orthogonal with gain=sqrt(2)
        - Shared FC layers: Orthogonal with gain=sqrt(2)
        - Actor head: Orthogonal with gain=0.01 (small initial policy)
        - Critic head: Orthogonal with gain=1.0
        - Biases: zeros
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for actor and critic heads
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)  # Small initial policy
        nn.init.constant_(self.actor_head.bias, 0)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.constant_(self.critic_head.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, 3, 6, 7)
               - Channel 0: Player 1 pieces
               - Channel 1: Player 2 pieces
               - Channel 2: Current player indicator
        
        Returns:
            Tuple containing:
                - action_probs: Action probability distribution, shape (batch_size, 7)
                                Probabilities sum to 1.0 for each state
                - state_value: State value estimate V(s), shape (batch_size, 1)
        
        Example:
            state = torch.randn(32, 3, 6, 7)  # Batch of 32 states
            action_probs, state_value = network(state)
            
            # Sample actions from policy
            actions = torch.multinomial(action_probs, 1)  # (32, 1)
            
            # Use values for advantage computation
            advantages = returns - state_value  # (32, 1)
        """
        # Shared convolutional layers
        for layer_dict in self.conv_layers:
            x = layer_dict['conv'](x)
            x = layer_dict['bn'](x)
            x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layers
        for layer_dict in self.shared_fc_layers:
            x = layer_dict['fc'](x)
            x = F.relu(x)
            if layer_dict['dropout'] is not None:
                x = layer_dict['dropout'](x)
        
        # Actor head: action logits → probabilities
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Critic head: state value
        state_value = self.critic_head(x)
        
        return action_probs, state_value
    
    def get_action_logits_and_value(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action logits (before softmax) and state value.
        
        Useful for computing log probabilities and entropy in training.
        
        Args:
            x: Input state tensor of shape (batch_size, 3, 6, 7)
        
        Returns:
            Tuple containing:
                - action_logits: Raw action logits, shape (batch_size, 7)
                - state_value: State value estimate V(s), shape (batch_size, 1)
        """
        # Shared convolutional layers
        for layer_dict in self.conv_layers:
            x = layer_dict['conv'](x)
            x = layer_dict['bn'](x)
            x = F.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Shared fully connected layers
        for layer_dict in self.shared_fc_layers:
            x = layer_dict['fc'](x)
            x = F.relu(x)
            if layer_dict['dropout'] is not None:
                x = layer_dict['dropout'](x)
        
        # Actor head: action logits (no softmax)
        action_logits = self.actor_head(x)
        
        # Critic head: state value
        state_value = self.critic_head(x)
        
        return action_logits, state_value
    
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
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'actor_parameters': sum(p.numel() for p in self.actor_head.parameters()),
            'critic_parameters': sum(p.numel() for p in self.critic_head.parameters())
        }
    
    def __repr__(self) -> str:
        """String representation of network."""
        config = self.get_config()
        return (f"ActorCriticNetwork(\n"
                f"  conv_channels={config['conv_channels']},\n"
                f"  fc_dims={config['fc_dims']},\n"
                f"  dropout={config['dropout_rate']},\n"
                f"  total_parameters={config['total_parameters']:,}\n"
                f"  actor_parameters={config['actor_parameters']:,}\n"
                f"  critic_parameters={config['critic_parameters']:,}\n"
                f")")