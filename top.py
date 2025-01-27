import argparse
import numpy as np
import logging
import csv
from ns3gym import ns3env
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Rule-Based Wi-Fi Slicing Simulation')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations, Default: 10')
    parser.add_argument('--log', type=str, default='rule_based_simulation_log.csv', help='Log file name, Default: rule_based_simulation_log.csv')
    parser.add_argument('--plot_dir', type=str, default='rule_based_plots', help='Directory to save plots')
    return parser.parse_args()

args = parse_arguments()
iteration_num = int(args.iterations)
log_file = args.log
plot_dir = args.plot_dir
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Simulation Parameters
PARAMS = {
    'channelNumber': 42, 'channelWidth': 20, 'mcs': 5, 'txPower': 15
}
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds

# Initialize the ns3 environment
env = ns3env.Ns3Env(
    port=5555, stepTime=STEP_TIME, startSim=True, simSeed=0, simArgs={"--simTime": SIM_TIME}, debug=True
)

# Reset environment and get action space
env.reset()
logging.info(f"Observation space: {env.observation_space}")
logging.info(f"Action space: {env.action_space}")

# Rule-Based Adjustments for Each Slice
def adjust_parameters(state, slice_type):
    if slice_type == "eMBB":
        if state["PacketLoss"] > 0.02:  # High packet loss
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Double bandwidth
        elif state["Throughput"] < 50:  # Low throughput
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase PTX
    elif slice_type == "mMTC":
        if state["PacketLoss"] > 0.02:  # High packet loss
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase PTX
        else:
            state["txPower"] = max(state["txPower"] - 1, 5)  # Save energy
    elif slice_type == "URLLC":
        if state["Latency"] > 5:  # High latency
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Increase bandwidth
        elif state["PacketLoss"] > 0.001:  # High packet loss
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase PTX
    return state

# Main rule-based slicing simulation loop
def rule_based_slicing():
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Slice", "Iteration", "Step", "Throughput", "Latency", "PacketLoss", 
                         "txPower", "channelWidth"])

        slices = ["eMBB", "mMTC", "URLLC"]

        for slice_type in slices:
            for iteration in range(iteration_num):
                state = PARAMS.copy()  # Initialize parameters
                done = False
                step_count = 0

                env.reset()

                while not done and step_count < 100:  # Limit steps per iteration
                    # Prepare action dictionary
                    action = {
                        "chNum": [state["channelNumber"]] * 3,
                        "mcs": [state["mcs"]] * 3,
                        "txPower": [state["txPower"]] * 3
                    }

                    try:
                        next_state, _, done, _ = env.step(action)
                    except KeyError as e:
                        logging.error(f"Error in action or step execution: {e}")
                        break

                    # Extract performance metrics from next_state
                    try:
                        throughput = next_state[slice_type][0][0]
                        latency = next_state[slice_type][0][1]
                        packet_loss = next_state[slice_type][0][2]
                    except (KeyError, IndexError) as e:
                        logging.error(f"Error extracting metrics: {e}")
                        continue

                    # Update state
                    state.update({"Throughput": throughput, "Latency": latency, "PacketLoss": packet_loss})

                    # Adjust parameters dynamically
                    state = adjust_parameters(state, slice_type)

                    # Log results
                    writer.writerow([slice_type, iteration + 1, step_count + 1, throughput, latency, packet_loss, 
                                     state["txPower"], state["channelWidth"]])

                    step_count += 1

    logging.info(f"Rule-based simulation results saved to {log_file}")

# Run the simulation
try:
    logging.info("Starting Rule-Based Dynamic Slicing Simulation...")
    rule_based_slicing()
finally:
    env.close()
    logging.info("Environment closed. Simulation complete.")

