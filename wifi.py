import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from ns3gym import ns3env
import logging
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='DQN Wi-Fi slicing simulation')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations, Default: 10')
    parser.add_argument('--log', type=str, default='simulation_dqn_log.csv', help='Log file name, Default: simulation_log.csv')
    parser.add_argument('--plot_dir', type=str, default='plots_dqn', help='Directory to save plots')
    return parser.parse_args()

args = parse_arguments()
iteration_num = int(args.iterations)
log_file = args.log
plot_dir = args.plot_dir
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000

# DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Environment setup
env = ns3env.Ns3Env(port=5555, stepTime=1.0, startSim=True, simSeed=0, simArgs={"--simTime": 20, "--testArg": 123}, debug=True)
state_dim = 60  # Adjust based on state flattening
action_dim = 3  # Adjust for the action space

# Initialize DQN and memory
q_network = DQN(state_dim, action_dim)
target_network = DQN(state_dim, action_dim)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)

# Function to preprocess state
def preprocess_state(state):
    """Flattens and converts state into a homogeneous numeric array."""
    flattened = []
    for slice_key in ['SliceA', 'SliceB', 'SliceC']:
        slice_data = state.get(slice_key, [])
        for item in slice_data:
            if isinstance(item, np.ndarray):
                # Flatten and convert ndarray to list
                flattened.extend(item.flatten().tolist())
            elif isinstance(item, (list, tuple)):
                # Flatten lists or tuples directly
                flattened.extend([float(sub_item) for sub_item in item if isinstance(sub_item, (int, float))])
            elif isinstance(item, (int, float)):
                # Directly append numbers
                flattened.append(float(item))
            else:
                logging.warning(f"Skipping non-numeric state element: {item}")
    return np.array(flattened, dtype=np.float32)

# Function to select action
def select_action(state):
    global EPSILON
    if random.random() < EPSILON:
        # Sample a random action as a dictionary with matching keys
        return {param: action_space.spaces[param].sample() for param in action_space.spaces.keys()}
    else:
        # Use the DQN model to predict the action
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = q_network(state)
        
        # Assuming a single action dimension, choose based on Q-values
        chosen_action = q_values.argmax().item()
        
        # Map chosen action back to action space structure as a dictionary
        return {param: chosen_action for param in action_space.spaces.keys()}

# Function to store experience
def store_experience(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Function to train the DQN
def train_dqn():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    current_q_values = q_network(states).gather(1, actions)
    max_next_q_values = target_network(next_states).max(1)[0]
    target_q_values = rewards + (GAMMA * max_next_q_values * (1 - dones))

    loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main training loop
for iteration in range(iteration_num):
    state = preprocess_state(env.reset())
    done = False
    episode_reward = 0

    while not done:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)
        next_state = preprocess_state(next_state)
        store_experience(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        train_dqn()

    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)
    target_network.load_state_dict(q_network.state_dict())
    logging.info(f"Iteration {iteration + 1}/{iteration_num}, Reward: {episode_reward}")

# Close the environment
env.close()

# Save results and plot if needed
logging.info("Simulation complete. Results saved.")

