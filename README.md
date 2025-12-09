# connect4-rl
Explore Reinforcement Learning via the connect 4 game.  Goal is to implement good policies using DQN, A2C, PPO and alpha-zero methods.

# Connect 4 Reinforcement Learning

A comprehensive implementation of multiple reinforcement learning algorithms applied to Connect 4.

## Algorithms Implemented ( or at least attempted ) 

- **DQN** (Deep Q-Network) - Value-based learning
- **A2C** (Advantage Actor-Critic) - Policy gradient method
- **PPO** (Proximal Policy Optimization) - Advanced policy gradient
- **AlphaZero** - MCTS + Neural Networks

## Project Structure
connect4-rl/
├── src/
│   ├── environment/     # Connect 4 game logic
│   ├── agents/          # RL agent implementations
│   ├── networks/        # Neural network architectures
│   ├── training/        # Training loops
│   ├── utils/           # Helper functions
│   └── gameplay/        # Human vs AI, Agent vs Agent
├── scripts/             # Training and demo scripts
├── tests/               # Unit tests
├── notebooks/           # Jupyter notebooks for analysis
├── models/              # Saved model checkpoints
├── logs/                # Training logs
└── results/             # Plots and results

## Setup

# Clone the repository
git clone https://github.com/ccolpitt/connect4-rl.git
cd connect4-rl

# Install dependencies
pip install -r requirements.txt

## Usage
Coming soon!

Development Roadmap
 Project structure
 Connect 4 environment
 DQN agent
 A2C agent
 PPO agent
 AlphaZero agent
 Training visualization
 Interactive gameplay