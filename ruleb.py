import argparse
import logging
import csv
from ns3gym import ns3env
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Argument parser
def parse_arguments():
    parser = argparse.ArgumentParser(description="Rule-Based Dynamic Slicing with ns3-gym")
    parser.add_argument("--start", type=int, default=1, help="Start ns-3 simulation (0/1, Default: 1)")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations (Default: 10)")
    parser.add_argument("--log", type=str, default="results_rule_based/ns3_dynamic_slicing_log.csv",
                        help="Log file name, Default: results_rule_based/ns3_dynamic_slicing_log.csv")
    return parser.parse_args()

# Parse arguments
args = parse_arguments()
start_sim = bool(args.start)
iteration_num = args.iterations
log_file = args.log

# Simulation Parameters
PORT = 0  # Use dynamic port selection
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds
SEED = 0
SIM_ARGS = {"--simTime": SIM_TIME}
DEBUG = True
MAX_STEPS = 100

# Initialize ns3-gym environment
def create_ns3_env(sim_args, start_sim=True):
    env = ns3env.Ns3Env(
        port=PORT, stepTime=STEP_TIME, startSim=start_sim, simSeed=SEED, simArgs=sim_args, debug=DEBUG
    )
    logging.info(f"Assigned port: {env.port}")
    logging.info(f"Action Space Structure: {env.action_space}")
    if not env or not hasattr(env, 'observation_space') or not hasattr(env, 'action_space'):
        raise ValueError("Failed to initialize ns3-gym environment. Check ns3 configuration.")
    return env

# Rule-Based Action Adjustments
def adjust_parameters(state, key):
    # Rule-based adjustments for each key or slice type
    if "eMBB" in key:
        if state["Pe"] > 0.02:  # High packet error
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Double bandwidth, cap at 80
        elif state["Throughput"] < 50:
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase txPower, cap at 20
    elif "mMTC" in key:
        if state["Pe"] > 0.02:
            state["txPower"] = min(state["txPower"] + 1, 20)  # Increase txPower
        else:
            state["txPower"] = max(state["txPower"] - 1, 5)  # Save power, floor at 5
    elif "URLLC" in key:
        if state["Latency"] > 5:
            state["channelWidth"] = min(state["channelWidth"] * 2, 80)  # Reduce latency, cap at 80
        elif state["Pe"] > 0.001:
            state["txPower"] = min(state["txPower"] + 1, 20)  # Improve reliability
    return state

# Training Loop
def rule_based_slicing(env, iterations=10, log_file="dynamic_results.csv"):
    """
    Rule-based dynamic slicing with ns3-gym.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Key", "Iteration", "Step", "Pe", "Latency", "Throughput", "txPower", "channelWidth"])

        for iteration in range(iterations):
            state = {"txPower": 20, "channelWidth": 20, "mcs": 5}  # Initial parameters
            total_reward = 0
            step_count = 0

            env.reset()
            done = False

            while not done and step_count < MAX_STEPS:
                # Match action keys to action space dynamically
                action = {}
                for key in env.action_space.spaces.keys():
                    if key == "chNum":
                        action[key] = [42, 42, 42]  # Provide values for all 3 dimensions
                    elif key == "mcs":
                        action[key] = [state["mcs"], state["mcs"], state["mcs"]]
                    elif key == "txPower":
                        action[key] = [state["txPower"], state["txPower"], state["txPower"]]

                try:
                    next_state, _, done, _ = env.step(action)
                except KeyError as e:
                    logging.error(f"Action space key error: {e}")
                    break

                # Log next_state structure
                logging.info(f"Next State Structure: {next_state}")

                # Dynamically handle all available keys in next_state
                for key, metrics in next_state.items():
                    try:
                        # Extract performance metrics
                        pe = metrics[0][0] if len(metrics[0]) > 0 else None
                        latency = metrics[0][1] if len(metrics[0]) > 1 else None
                        throughput = metrics[0][2] if len(metrics[0]) > 2 else None
                    except (KeyError, IndexError) as e:
                        logging.error(f"Error extracting metrics for key {key}: {e}")
                        continue

                    # Update state with observed metrics
                    state.update({"Pe": pe, "Latency": latency, "Throughput": throughput})

                    # Adjust parameters based on observed metrics
                    state = adjust_parameters(state, key)

                    # Log metrics
                    writer.writerow([key, iteration + 1, step_count + 1, pe, latency, throughput, state["txPower"], state["channelWidth"]])

                step_count += 1

    logging.info(f"Rule-Based Slicing Results saved to {log_file}")

# Main Execution
def main():
    env = create_ns3_env(SIM_ARGS, start_sim=start_sim)

    # Rule-Based Slicing
    logging.info("Starting Rule-Based Dynamic Slicing...")
    rule_based_slicing(env, iterations=iteration_num, log_file=log_file)

if __name__ == "__main__":
    main()

