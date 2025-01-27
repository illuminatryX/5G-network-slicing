#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from ns3gym import ns3env
from gym.spaces import flatten_space, flatten, unflatten

# Setup logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description='Start RL-based Wi-Fi slicing simulation')
    parser.add_argument('--start', type=int, default=1, help='Start ns-3 simulation script 0/1, Default: 1')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations, Default: 10')
    parser.add_argument('--log', type=str, default='simulation_log.csv', help='Log file name, Default: simulation_log.csv')
    return parser.parse_args()

# Initialize and parse command-line arguments
args = parse_arguments()
start_sim = bool(args.start)
iteration_num = int(args.iterations)
log_file = args.log

# Simulation Parameters
PARAMS = {
    'channelNumber': 42, 'channelWidth': 80, 'gi': 800, 'mcs': 5, 'txPower': 20
}
PORT = 0
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds
SEED = 0
SIM_ARGS = {"--simTime": SIM_TIME, "--testArg": 123}
DEBUG = True

# Initialize the ns3 environment
try:
    env = ns3env.Ns3Env(
        port=PORT, stepTime=STEP_TIME, startSim=start_sim, simSeed=SEED, simArgs=SIM_ARGS, debug=DEBUG
    )
    logging.info("Environment initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize environment: {e}")
    exit(1)

# Reset the environment
try:
    env.reset()
except Exception as e:
    logging.error(f"Failed to reset environment: {e}")
    exit(1)

# Extract observation and action spaces
obs_space = env.observation_space
action_space = env.action_space

if obs_space is None or action_space is None:
    logging.error("Observation or action space is None. Check ns3 environment initialization.")
    exit(1)

# Determine state dimensions
try:
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        state_dim = obs_space.shape[0]
    elif hasattr(obs_space, 'spaces'):
        # Handle Dict observation space
        state_dim = sum(space.shape[0] if hasattr(space, 'shape') else 1 for space in obs_space.spaces.values())
    else:
        raise ValueError("Unsupported observation space format.")
except Exception as e:
    logging.error(f"Failed to determine state dimensions: {e}")
    exit(1)

# Flatten the action space
flat_action_space = flatten_space(action_space)
action_dim = flat_action_space.shape[0]

logging.info(f"Observation space dimensions: {state_dim}")
logging.info(f"Flattened action space dimension: {action_dim}")

# Replay Buffer
replay_buffer = deque(maxlen=10000)

def store_transition(state, action, reward, next_state, done):
    replay_buffer.append((state, action, reward, next_state, done))

# DQN Model
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

def train_dqn(model, target_model, optimizer, batch_size=64, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    
    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32)

    # Compute Q-values
    q_values = model(states).gather(1, actions.long().unsqueeze(1)).squeeze()
    with torch.no_grad():
        max_next_q_values = target_model(next_states).max(1)[0]
    targets = rewards + gamma * max_next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Sample a random action
def sample_action(action_space):
    """Samples a random action from a Dict action space."""
    return {key: space.sample() for key, space in action_space.spaces.items()}

# Map a flattened action back to Dict
def map_action_from_flat(flat_action):
    """Maps a flattened action vector back to the original Dict action space."""
    return unflatten(action_space, flat_action)

def log_metrics(iteration, step, reward):
    """Logs simulation metrics to a CSV file."""
    with open(log_file, 'a') as f:
        f.write(f"{iteration},{step},{reward}\n")

def main():
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    epsilon = 1.0  # Exploration rate
    epsilon_decay = 0.995
    min_epsilon = 0.01
    gamma = 0.99
    update_target_steps = 100
    step_count = 0

    for curr_it in range(iteration_num):
        state = env.reset()
        done = False

        while not done:
            step_count += 1
            
            # Epsilon-greedy policy
            if random.random() < epsilon:
                action = sample_action(action_space)
            else:
                flat_state = torch.tensor(state, dtype=torch.float32)
                flat_action = torch.argmax(policy_net(flat_state)).item()
                action = map_action_from_flat(flat_action)

            # Interact with the environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            flat_action = flatten(action_space, action)
            store_transition(state, flat_action, reward, next_state, done)
            
            # Train DQN
            train_dqn(policy_net, target_net, optimizer)

            state = next_state

            # Log metrics
            log_metrics(curr_it + 1, step_count, reward)

            # Update target network periodically
            if step_count % update_target_steps == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()
    logging.info("Simulation finished")
    logging.info("Results saved to log file.")

if __name__ == "__main__":
    main()

