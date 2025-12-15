import torch
import torch.nn as nn
import torch.optim as optim

# 1. Simple Dummy Network (Linear for clarity)
class SimpleDQN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 2 channels, 6 rows, 7 cols = 84 inputs
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 50),
            nn.ReLU(),
            nn.Linear(50, 7) # Output 7 column Q-values
        )

    def forward(self, x):
        return self.fc(x)

def test_single_example_convergence():
    net = SimpleDQN()
    optimizer = optim.Adam(net.parameters(), lr=0.01) # High LR for quick check
    loss_fn = nn.MSELoss()

    # 2. Create ONE dummy winning transition
    # State: Random board
    state = torch.randn(1, 2, 6, 7) 
    
    # Action: Column 3
    action = torch.tensor([[3]]) 
    
    # Reward: +1 (We won)
    reward = torch.tensor([[1.0]])
    
    # Next State: Random (Should be ignored because done=True)
    next_state = torch.randn(1, 2, 6, 7)
    
    # Done: TRUE
    done = torch.tensor([[1.0]]) 
    gamma = 0.99

    print("--- Starting Single Example Overfit Test ---")
    
    for i in range(50):
        # 3. Forward Pass
        q_values = net(state) # Shape [1, 7]
        
        # 4. Select Q-value for action taken
        # gather: select value at index 3
        q_value = q_values.gather(1, action) 

        # 5. Calculate Target (Negamax Logic)
        with torch.no_grad():
            # Get max q of next state (opponent's turn)
            next_q_values = net(next_state)
            best_next_q = next_q_values.max(1)[0].unsqueeze(1)
            
            # THE FORMULA: r - gamma * max_next * (1 - done)
            # If done is 1, the second term becomes 0. Target = r.
            target = reward - (gamma * best_next_q * (1 - done))

        # 6. Loss
        loss = loss_fn(q_value, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"Iter {i}: Q-val: {q_value.item():.4f}, Target: {target.item():.4f}, Loss: {loss.item():.6f}")

    print(f"Final Q-value for Action 3: {q_value.item():.4f} (Should be close to 1.0)")

test_single_example_convergence()