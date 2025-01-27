#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import csv
from ns3gym import ns3env
from gym.spaces import Box, Dict

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Slicing vs Single-Band with ns3-gym")
    parser.add_argument("--start", type=int, default=1, help="Start ns-3 simulation (0/1, Default: 1)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations (Default: 10)")
    parser.add_argument("--log_dynamic", type=str, default="dynamic_results.csv", help="Log file for dynamic slicing (Default: dynamic_results.csv)")
    parser.add_argument("--log_static", type=str, default="static_results.csv", help="Log file for single-band (Default: static_results.csv)")
    return parser.parse_args()

# Parse command-line arguments
args = parse_arguments()
start_sim = bool(args.start)
iteration_num = int(args.iterations)
log_dynamic = args.log_dynamic
log_static = args.log_static

# Simulation Parameters
PORT = 0  # Use dynamic port selection
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds
SEED = 0
SIM_ARGS = {"--simTime": SIM_TIME, "--testArg": 123}
DEBUG = True

# Initialize ns3-gym environment
def create_ns3_env(sim_args, start_sim=True):
    env = ns3env.Ns3Env(
        port=PORT, stepTime=STEP_TIME, startSim=start_sim, simSeed=SEED, simArgs=sim_args, debug=DEBUG
    )
    logging.info(f"Assigned port: {env.port}")
    if not env or not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
        raise ValueError("Failed to initialize ns3-gym environment. Check ns3 configuration.")
    return env

# Define DQN Model
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

# Flatten Observation
def flatten_observation(obs_dict):
    """
    Flattens a nested observation dictionary into a single list.

    Args:
        obs_dict (dict): The observation dictionary.

    Returns:
        list: Flattened observation.
    """
    flat_obs = []
    for key, value in obs_dict.items():
        if isinstance(value, tuple):  # Handle Tuple in Dict
            for box in value:
                if hasattr(box, "__iter__"):
                    flat_obs.extend([float(x) for x in box])  # Ensure all elements are floats
                else:
                    flat_obs.append(float(box))
        elif isinstance(value, (list, np.ndarray)):  # Handle list or array
            flat_obs.extend([float(x) for x in value])
        elif isinstance(value, (int, float)):  # Handle scalar values
            flat_obs.append(float(value))
        elif hasattr(value, "__iter__"):  # Handle other iterable types (e.g., RepeatedScalarContainer)
            flat_obs.extend([float(x) for x in value])
        else:
            raise TypeError(f"Unsupported observation type: {type(value)} in key {key}")
    logging.info(f"Flattened Observation: {flat_obs}")
    return flat_obs

# Train Dynamic Slicing with DQN
def train_dqn(env, iterations=10, gamma=0.99, batch_size=64, lr=1e-3, epsilon_decay=0.995):
    # Debug observation and action spaces
    logging.info(f"Observation Space: {env.observation_space}")
    logging.info(f"Action Space: {env.action_space}")

    # Flattened observation and action dimensions
    sample_obs = env.reset()
    state_dim = len(flatten_observation(sample_obs))
    action_dim = sum([space.shape[0] for space in env.action_space.spaces.values()])

    logging.info(f"Flattened State Dimension: {state_dim}, Action Dimension: {action_dim}")

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)

    epsilon = 1.0
    results = []

    for iteration in range(iterations):
        raw_state = env.reset()
        state = flatten_observation(raw_state)
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            if random.random() < epsilon:
                action = {key: space.sample() for key, space in env.action_space.spaces.items()}
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action_index = policy_net(state_tensor).argmax().item()
                    action = {key: int(action_index % space.shape[0]) for key, space in env.action_space.spaces.items()}

            raw_next_state, reward, done, info = env.step(action)
            next_state = flatten_observation(raw_next_state)

            memory.append((state, action, reward, next_state, done))
            total_reward += reward
            step_count += 1
            state = next_state

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = policy_net(states)
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, target_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(0.01, epsilon * epsilon_decay)
        results.append([iteration + 1, step_count, total_reward])
        logging.info(f"DQN Iteration {iteration + 1}: Total Reward: {total_reward}")

    env.close()
    return results

# Main Execution
def main():
    # Initialize ns3-gym environment
    env_dynamic = create_ns3_env(SIM_ARGS, start_sim=start_sim)

    # Dynamic Slicing (DQN)
    logging.info("Starting Dynamic Slicing with DQN...")
    dynamic_results = train_dqn(env_dynamic, iterations=iteration_num)

    # Save results
    with open(log_dynamic, "w", newline="") as f_dynamic:
        writer = csv.writer(f_dynamic)
        writer.writerow(["Iteration", "Step", "Total_Reward"])
        writer.writerows(dynamic_results)

if __name__ == "__main__":
    main()

