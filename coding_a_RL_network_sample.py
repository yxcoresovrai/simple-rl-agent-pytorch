# Torch libraries
import torch
import torch.nn as nn
import torch.optim as optim

# Random, Numpy, and Deque
import random 
import numpy as np
from collections import deque 

# === Sample Data ===
state_dim = 4 # Assume 4 state features
action_dim = 2 # Assume 2 possible actions

# === Generate sample data for 10 episodes, each with 200 steps
num_episodes = 10
episode_length = 200 

sample_data = [] 

for _ in range(num_episodes):
    episode_data = []
    state = np.random.rand(state_dim) 
    for _ in range(episode_length): 
        action = random.randint(0, action_dim - 1) 
        reward = random.uniform(0, 1)   # Random reward between 0 and 1
        next_state = np.random.rand(state_dim) 
        done = random.random() < 0.1    # 10% chance of episode ending 
        episode_data.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break 

    sample_data.append(episode_data)

# === DQN RL Network ===
class DQN(nn.Module):
    def __init__(self):
        super().__init__() 
        self.model = nn.Sequential(

            # nn.Linear applies a linear transformation. It performs y = xA^T + b, x = input, A = weight matrix, b = bias vector, y = output
            nn.Linear(state_dim, 128),

            # nn.ReLU applies the rectified linear unit function element-wise, y = max(0, x), x = input, y = output
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    # forward() method defines the computation performed at every call, forward pass of the model
    def forward(self, x):
        return self.model(x)
    
q_net = DQN()
# The optimizer is the algorithm that adjusts the weights of the network to minimize the loss function
optimizer = optim.Adam(q_net.parameters(), lr=1e-3)

# The criterion (nn.MSELoss()) is the loss function that measures the difference between the predicted value and the actual values
criterion = nn.MSELoss()

# The replay buffer stores the agent's experiences, which are used to train the network
replay_buffer = deque(maxlen=10000)

# The discount factor gamma determines the importance of future rewards
gamma = 0.99

# === Training Loop
for episode_data in sample_data:

    # total_reward keeps track of the total reward accumulated in the episode
    total_reward = 0

    # Loop through the episode steps
    for state, action, reward, next_state, done in episode_data:
        # Append state, action ,reward, next_state, done to replay_buffer (the agent's experiences)
        replay_buffer.append((state, action, reward, next_state, done))
        # Accumulate the reward to the total_reward 
        total_reward += reward

        # The agent samples a batch of experiences from the replay buffer and uses them to train the network
        if len(replay_buffer) >= 64:

            # random.sample() returns a random sample of items from the replay buffer
            batch = random.sample(replay_buffer, 64)

            # zip() groups the elements of the batch into states, actions, rewards, next_states, and dones with a tuple for each experience via zip(*batch)
            states, actions, rewards, next_states, dones = zip(*batch)

            # torch.FloatTensor() converts the states data to PyTorch tensors
            states = torch.FloatTensor(states) 

            # torch.LongTensor() converts the actions data to PyTorchTensor, and unsqueeze(1) adds a dimension to the tensor to match the actions tensor
            actions = torch.LongTensor(actions).unsqueeze(1) 

            # torch.FloatTensor() converts the rewards data to PyTorch tensors, and unsqueeze(1) adds a dimension to the tensor to match the actions tensor
            rewards = torch.FloatTensor(rewards).unsqueeze(1) 

            # torch.FloatTensor() converts the next_states data to PyTorch tensors 
            next_states = torch.FloatTensor(next_states) 

            # torch.FloatTensor() converts the dones data to PyTorch tensors, and unsqueeze(1) adds a dimension to the tensor to match the actions tensor
            dones = torch.FloatTensor(dones).unsqueeze(1) 

            # q_values are the predicted values for the actions in the current states, where q_net(states) returns the predicted values for all actions in the states,
            # and gather(1, actions) selects the predicted values for the actions in the states, 1 is the dimension along which to index, actions are the indices
            q_values = q_net(states).gather(1, actions)

            # next_q are the predicted values for the actions in the next states, where q_net(next_states) returns the predicted values for all actions in the next states
            # and .max(1, keepdim=True)[0].detach() returns the maximum predicted value for each next state, detach() detaches the tensor from the computation graph
            next_q = q_net(next_states).max(1, keepdim=True)[0].detach()

            # target_q are the target values for the actions in the current states, with rewards = rewards, gamma = discount factor, next_q = next_q, and dones = dones
            # target_q calculates the target value for the actions in the current states using the Bellman equation, where target_q = rewards + gamma * next_q * (1 - dones). Gamma is the discount factor, which determines the importance of future rewards. (1 - dones) represents the terminal state, where dones is a binary flag indicating if the episode is done.
            target_q = rewards + gamma * next_q * (1 - dones)

            # The loss is the difference between the predicted values (q_values) and the target values (target_q)
            loss = criterion(q_values, target_q)

            # The optimizer adjusts the weights of the network to minimize the loss, .zero_grad() clears the gradients, .backward() computes the gradients, and .step() updates the weights
            optimizer.zero_grad()

            # The loss.backward() computes the gradients of the loss with respect to the network parameters
            loss.backward()

            # The optimizer.step() updates the network parameters based on the gradients
            optimizer.step()

    print(f"Episode: Total Reward: {total_reward}")