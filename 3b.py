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
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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

# Define the Environment
class SlicingEnvironment:
    def __init__(self, slice_type):
        self.slice_type = slice_type
        self.state = {
            "Pe": np.random.uniform(0.0005, 0.02),
            "Latency": np.random.uniform(1, 10),
            "Throughput": np.random.uniform(50, 100) if slice_type == "eMBB" else \
                          np.random.uniform(10, 50) if slice_type == "mMTC" else \
                          np.random.uniform(20, 40),
            "PTX": 20,
            "Bw": 20,
        }
        self.step_count = 0

    def reset(self):
        self.state = {
            "Pe": np.random.uniform(0.0005, 0.02),
            "Latency": np.random.uniform(1, 10),
            "Throughput": np.random.uniform(50, 100) if self.slice_type == "eMBB" else \
                          np.random.uniform(10, 50) if self.slice_type == "mMTC" else \
                          np.random.uniform(20, 40),
            "PTX": 20,
            "Bw": 20,
        }
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return [
            self.state["Pe"],
            self.state["Latency"],
            self.state["Throughput"],
            self.state["PTX"],
            self.state["Bw"],
        ]

    def step(self, action):
        # Actions: 0=Increase PTX, 1=Decrease PTX, 2=Increase Bw, 3=Decrease Bw
        if action == 0:
            self.state["PTX"] += 1
        elif action == 1:
            self.state["PTX"] -= 1
        elif action == 2:
            self.state["Bw"] *= 2
        elif action == 3:
            self.state["Bw"] /= 2

        # Simulate network response
        self.state["Pe"] = np.random.uniform(0.0005, 0.03)
        self.state["Latency"] = np.random.uniform(1, 10)
        self.state["Throughput"] = np.random.uniform(50, 100)

        # Compute reward
        reward = self.compute_reward()
        self.step_count += 1
        done = self.step_count >= 100  # Terminate after 100 steps
        return self._get_state(), reward, done

    def compute_reward(self):
        Pe = self.state["Pe"]
        latency = self.state["Latency"]
        throughput = self.state["Throughput"]
        PTX = self.state["PTX"]

        if self.slice_type == "eMBB":
            return throughput - latency * 0.1 if Pe < 0.001 else -100
        elif self.slice_type == "mMTC":
            return -PTX if Pe <= 0.02 else -100
        elif self.slice_type == "URLLC":
            return 100 / latency if Pe < 0.001 and latency < 5 else -100

# Helper Functions
def select_action(state, policy_net, epsilon, action_space):
    if random.random() < epsilon:
        return random.randint(0, action_space - 1)
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state_tensor)
        return q_values.argmax().item()

def optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size):
    if len(memory) < batch_size:
        return
    transitions = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*transitions)

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

# Main Training Loop
def train_dqn(slice_type, iterations=10, gamma=0.99, batch_size=64, lr=1e-3, epsilon_decay=0.995):
    env = SlicingEnvironment(slice_type)
    policy_net = DQN(5, 4)
    target_net = DQN(5, 4)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=10000)

    epsilon = 1.0
    results_dir = "results_dqn"
    os.makedirs(results_dir, exist_ok=True)
    log_file = f"{results_dir}/{slice_type}_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Step", "Reward", "Pe", "Latency", "Throughput", "PTX", "Bw"])

        for iteration in range(iterations):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = select_action(state, policy_net, epsilon, 4)
                next_state, reward, done = env.step(action)
                memory.append((state, action, reward, next_state, done))
                optimize_model(memory, policy_net, target_net, optimizer, gamma, batch_size)
                state = next_state
                total_reward += reward
                writer.writerow([iteration + 1, env.step_count, reward, *state])
            
            epsilon = max(0.01, epsilon * epsilon_decay)
            logging.info(f"Iteration {iteration + 1}, Total Reward: {total_reward}")

# Run Training
for slice_type in ["eMBB", "mMTC", "URLLC"]:
    train_dqn(slice_type)

