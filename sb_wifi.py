#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import matplotlib.pyplot as plt
from ns3gym import ns3env
import logging
import os
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Single-Band Wi-Fi Simulation")
    parser.add_argument("--start", type=int, default=1, help="Start ns-3 simulation script 0/1, Default: 1")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations, Default: 10")
    parser.add_argument("--log", type=str, default="results_dqn/single_band_log.csv",
                        help="Log file name, Default: results_dqn/single_band_log.csv")
    return parser.parse_args()

args = parse_arguments()
start_sim = bool(args.start)
iteration_num = args.iterations
log_file = args.log

# Simulation Parameters
PORT = 5555
SIM_TIME = 20
STEP_TIME = 1.0
SEED = 0
SIM_ARGS = {
    "--simTime": SIM_TIME,
    "--band": "AC_5",  # Wi-Fi 5 (Single Band)
    "--channelNumberA": 36,
    "--channelWidthA": 40,  # Wider channel for compatibility
    "--mcsA": 5,
    "--giA": 800,  # Valid Guard Interval
    "--txPowerA": 20
}
DEBUG = True

# Results directory
results_dir = "results_dqn"
os.makedirs(results_dir, exist_ok=True)

# Initialize environment dynamically
def create_env(sim_args):
    env = ns3env.Ns3Env(
        port=0, stepTime=STEP_TIME, startSim=start_sim, simSeed=SEED, simArgs=sim_args, debug=DEBUG
    )
    assigned_port = env.port
    logging.info(f"Assigned Port: {assigned_port}")
    return env

# Create the environment
env = create_env(SIM_ARGS)

# Single-Band Wi-Fi Simulation
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["iteration", "step", "reward", "latency", "tx_packets", "rx_packets",
                     "channelNumberA", "channelWidthA", "giA", "mcsA", "txPowerA"])
    
    for iteration in range(iteration_num):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            action = env.action_space.sample()  # Random action for demonstration
            next_state, reward, done, info = env.step(action)
            latency = next_state.get("latency", 0)
            tx_packets = next_state.get("tx_packets", 0)
            rx_packets = next_state.get("rx_packets", 0)

            # Save metrics
            writer.writerow([iteration + 1, step + 1, reward, latency, tx_packets, rx_packets,
                             SIM_ARGS["--channelNumberA"], SIM_ARGS["--channelWidthA"], SIM_ARGS["--giA"],
                             SIM_ARGS["--mcsA"], SIM_ARGS["--txPowerA"]])
            total_reward += reward
            step += 1

        logging.info(f"Iteration {iteration + 1}: Total Reward: {total_reward}")

logging.info("Single-Band Wi-Fi Simulation Finished. Results saved in 'results_dqn'")

