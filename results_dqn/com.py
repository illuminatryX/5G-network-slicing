#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
import csv
from ns3gym import ns3env

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Simulation Parameters
PORT = 5555
STEP_TIME = 1.0
SIM_TIME = 20
SEED = 0
DEBUG = True

# Dynamic Simulation Arguments (Customized for Slicing)
BASE_SIM_ARGS = {
    "--simTime": SIM_TIME,
    "--band": "AX_5",
    "--channelNumberA": 36,
    "--channelWidthA": 80,
    "--mcsA": 5,
    "--giA": 800,
    "--txPowerA": 20,
}

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

# Initialize ns-3 environment
def create_ns3_env(sim_args, start_sim=True):
    env = ns3env.Ns3Env(
        port=PORT, stepTime=STEP_TIME, startSim=start_sim, simSeed=SEED, simArgs=sim_args, debug=DEBUG
    )
    logging.info(f"ns-3 Environment created on Port {PORT}")
    return env

# Train Dynamic Slicing with DQN
def train_dqn(slice_type, sim_args, iterations=10, gamma=0.99, batch_size=64, lr=1e-3, epsilon_decay=0.995):
    env = create_ns3_env(sim_args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQN setup
    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)

    epsilon = 1.0
    results = []

    for iteration in range(iterations):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action = policy_net(state_tensor).argmax().item()

            # Step in the environment
            next_state, reward, done, info = env.step(action)
            memory.append((state, action, reward, next_state, done))
            total_reward += reward
            step_count += 1

            # Optimize the policy
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones)

                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].detach()
                target_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = nn.MSELoss()(q_values, target_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state

        epsilon = max(0.01, epsilon * epsilon_decay)
        results.append([slice_type, iteration + 1, step_count, total_reward])
        logging.info(f"Slice: {slice_type}, Iteration: {iteration + 1}, Total Reward: {total_reward}")

    env.close()
    return results

# Run Single-Band Static Configuration
def train_static(slice_type, sim_args, iterations=10):
    sim_args["--dynamic"] = "false"  # Disable dynamic adjustments
    env = create_ns3_env(sim_args, start_sim=True)
    results = []

    for iteration in range(iterations):
        state = env.reset()
        total_reward = 0
        done = False
        step_count = 0

        while not done:
            # Static approach uses a fixed action (e.g., maintain baseline)
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            step_count += 1
            state = next_state

        results.append([slice_type, iteration + 1, step_count, total_reward])
        logging.info(f"Slice: {slice_type}, Iteration: {iteration + 1}, Total Reward: {total_reward}")

    env.close()
    return results

# Run both dynamic slicing and static configuration
def main():
    slices = ["eMBB", "mMTC", "URLLC"]
    dynamic_results = []
    static_results = []

    for slice_type in slices:
        # Customize simulation arguments for each slice
        sim_args = BASE_SIM_ARGS.copy()
        if slice_type == "eMBB":
            sim_args["--channelWidthA"] = 80
            sim_args["--txPowerA"] = 20
        elif slice_type == "mMTC":
            sim_args["--channelWidthA"] = 20
            sim_args["--txPowerA"] = 10
        elif slice_type == "URLLC":
            sim_args["--channelWidthA"] = 40
            sim_args["--txPowerA"] = 20

        # Train DQN-based dynamic slicing
        dynamic_results.extend(train_dqn(slice_type, sim_args))

        # Train single-band static configuration
        static_results.extend(train_static(slice_type, sim_args))

    # Save results to CSV
    with open("dynamic_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Slice", "Iteration", "Step", "Total_Reward"])
        writer.writerows(dynamic_results)

    with open("static_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Slice", "Iteration", "Step", "Total_Reward"])
        writer.writerows(static_results)

if __name__ == "__main__":
    main()

