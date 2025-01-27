import argparse
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
    parser.add_argument('--log', type=str, default='rule_based_simulation_log.csv', help='Log file name')
    return parser.parse_args()

args = parse_arguments()
iteration_num = args.iterations
log_file = args.log
log_dir = os.path.dirname(log_file) or "."
os.makedirs(log_dir, exist_ok=True)

# Simulation Parameters
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds
SIM_ARGS = {"--simTime": SIM_TIME}  # Custom arguments can be added here
DEBUG = True
MAX_STEPS = 100

# Slice name mapping
slice_mapping = {
    "eMBB": "SliceA",
    "mMTC": "SliceB",
    "URLLC": "SliceC"
}

# Initialize the ns3 environment with dynamic port assignment
env = ns3env.Ns3Env(
    port=0, stepTime=STEP_TIME, startSim=True, simSeed=0, simArgs=SIM_ARGS, debug=True
)

logging.info(f"Action space structure: {env.action_space}")

# Rule-Based Adjustments for Each Slice
def adjust_parameters(state, slice_type):
    if slice_type == "eMBB":
        if state["Latency"] > 50:  # High latency
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Double bandwidth
        elif state["Throughput"] < 50:  # Low throughput
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase PTX
    elif slice_type == "mMTC":
        if state["Latency"] > 100:  # High latency
            state["channelWidth"] = min(state["channelWidth"] * 2, 40)  # Increase bandwidth for IoT devices
        else:
            state["txPower"] = max(state["txPower"] - 1, 5)  # Save energy
    elif slice_type == "URLLC":
        if state["Latency"] > 5:  # High latency threshold
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Double bandwidth
            state["txPower"] = 20  # Maximize transmission power
        elif state["PacketLoss"] > 0.001:  # High packet loss
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase PTX
    return state

# Main rule-based slicing simulation loop
def rule_based_slicing():
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Slice", "Iteration", "Step", "Throughput", "Latency", "PacketLoss", 
                         "txPower", "channelWidth"])

        slices = ["eMBB", "mMTC", "URLLC"]

        for slice_type in slices:
            for iteration in range(iteration_num):
                state = {"channelWidth": 20, "txPower": 15, "mcs": 5}
                done = False
                step_count = 0

                env.reset()

                while not done and step_count < MAX_STEPS:
                    # Prepare action dynamically
                    action = {
                        key: env.action_space.spaces[key].sample()
                        for key in env.action_space.spaces.keys()
                    }

                    try:
                        next_state, _, done, _ = env.step(action)
                        logging.info(f"Next state: {next_state}")
                    except Exception as e:
                        logging.error(f"Error in step execution: {e}")
                        break

                    # Map slice type to next_state key
                    mapped_key = slice_mapping.get(slice_type)
                    if mapped_key in next_state:
                        metrics = next_state[mapped_key]
                        avg_throughput = sum(metrics[0]) / len(metrics[0])  # Average throughput
                        avg_latency = sum(metrics[1]) / len(metrics[1]) / 10  # Scale latency by dividing by 10
                        avg_packet_loss = sum(metrics[2]) / len(metrics[2])  # Average packet loss
                    else:
                        logging.error(f"Slice {slice_type} (mapped to {mapped_key}) not found in next_state")
                        continue

                    # Update state and adjust parameters
                    state.update({"Throughput": avg_throughput, "Latency": avg_latency, "PacketLoss": avg_packet_loss})
                    state = adjust_parameters(state, slice_type)

                    # Log results
                    writer.writerow([slice_type, iteration + 1, step_count + 1, avg_throughput, avg_latency, avg_packet_loss, 
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

