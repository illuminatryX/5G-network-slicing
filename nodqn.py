import argparse
import numpy as np
import logging
import os
import csv

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Dynamic Slicing (Rule-Based)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations, Default: 10")
    parser.add_argument("--log", type=str, default="results_rule_based/dynamic_slicing_log.csv",
                        help="Log file name, Default: results_rule_based/dynamic_slicing_log.csv")
    return parser.parse_args()

args = parse_arguments()
iteration_num = args.iterations
log_file = args.log

# Simulation Parameters
MAX_STEPS = 100

# Action Definitions (Rule-Based)
def adjust_parameters(state, slice_type):
    # Rule-based adjustments
    if slice_type == "eMBB":
        if state["Pe"] > 0.02:  # High packet error
            state["Bw"] *= 2  # Double bandwidth
        elif state["Throughput"] < 50:
            state["PTX"] += 1  # Increase transmission power
    elif slice_type == "mMTC":
        if state["Pe"] > 0.02:
            state["PTX"] += 1  # Increase PTX
        else:
            state["PTX"] -= 1  # Save power
    elif slice_type == "URLLC":
        if state["Latency"] > 5:
            state["Bw"] *= 2  # Reduce latency with more bandwidth
        elif state["Pe"] > 0.001:
            state["PTX"] += 1  # Improve reliability with higher PTX
    return state

# Reward Function
def compute_reward(slice_type, Pe, latency, throughput, PTX, Bw):
    if slice_type == "eMBB":
        if Pe < 0.001:
            reward = throughput - latency * 0.1
        else:
            reward = -100  # Penalize high error rates
    elif slice_type == "mMTC":
        if Pe <= 0.02 and PTX < 20:
            reward = -PTX  # Reward energy saving
        else:
            reward = -100  # Penalize high error rates
    elif slice_type == "URLLC":
        if Pe < 0.001 and latency < 5:
            reward = 100 / latency  # Reward low latency
        else:
            reward = -100  # Penalize high error rates or latency
    return reward

# Simulation Logic
def simulate_network_response(slice_type, state):
    Pe = np.random.uniform(0.0005, 0.03)  # Simulate random Pe
    latency = np.random.uniform(1, 10)   # Simulate random latency
    throughput = np.random.uniform(10, 100) if slice_type == "eMBB" else \
                 np.random.uniform(5, 50) if slice_type == "mMTC" else \
                 np.random.uniform(20, 40)
    return Pe, latency, throughput

# Training Loop
os.makedirs(os.path.dirname(log_file), exist_ok=True)
with open(log_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Slice", "Iteration", "Step", "Reward", "Pe", "Latency", "Throughput", "PTX", "Bw"])

    for slice_type in ["eMBB", "mMTC", "URLLC"]:
        for iteration in range(iteration_num):
            state = {"PTX": 20, "MCS": 5, "Bw": 20, "GI": 800}
            total_reward = 0

            for step_count in range(MAX_STEPS):
                # Simulate network response
                Pe, latency, throughput = simulate_network_response(slice_type, state)
                state["Pe"] = Pe
                state["Latency"] = latency
                state["Throughput"] = throughput

                # Adjust parameters using rule-based logic
                state = adjust_parameters(state, slice_type)

                # Compute reward
                reward = compute_reward(slice_type, Pe, latency, throughput, state["PTX"], state["Bw"])
                total_reward += reward

                writer.writerow([slice_type, iteration + 1, step_count + 1, reward, Pe, latency, throughput, state["PTX"], state["Bw"]])

            logging.info(f"Slice: {slice_type}, Iteration: {iteration + 1}, Total Reward: {total_reward}")

logging.info("Dynamic Slicing Simulation Finished.")

