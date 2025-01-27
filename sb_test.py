import argparse
import logging
import csv
from ns3gym import ns3env
import os

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Argument parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description='Single-Band Wi-Fi Simulation')
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations, Default: 10')
    parser.add_argument('--log', type=str, default='single_band_simulation_log.csv', help='Log file name')
    return parser.parse_args()

args = parse_arguments()
iteration_num = args.iterations
log_file = args.log
log_dir = os.path.dirname(log_file) or "."
os.makedirs(log_dir, exist_ok=True)

# Simulation Parameters
SIM_TIME = 20  # seconds
STEP_TIME = 1.0  # seconds
SIM_ARGS = {"--simTime": SIM_TIME}  # Basic single-band configuration
DEBUG = True
MAX_STEPS = 100

# Initialize the ns3 environment
env = ns3env.Ns3Env(
    port=0, stepTime=STEP_TIME, startSim=True, simSeed=0, simArgs=SIM_ARGS, debug=True
)

logging.info(f"Action space structure: {env.action_space}")

# Main single-band simulation loop
def single_band_simulation():
    with open(log_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Iteration", "Step", "Throughput", "Latency", "PacketLoss", "txPower", "channelWidth"])

        for iteration in range(iteration_num):
            state = {"channelWidth": 20, "txPower": 15, "mcs": 5}  # Fixed initial parameters for all devices
            done = False
            step_count = 0

            env.reset()

            while not done and step_count < MAX_STEPS:
                # Prepare action
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

                # Aggregate metrics across all slices
                try:
                    throughputs = []
                    latencies = []
                    packet_losses = []

                    for slice_key in ["SliceA", "SliceB", "SliceC"]:
                        if slice_key in next_state:
                            metrics = next_state[slice_key]
                            throughputs.extend(metrics[0])  # Aggregate throughput
                            latencies.extend(metrics[1])    # Aggregate latency
                            packet_losses.extend(metrics[2])  # Aggregate packet loss

                    # Compute averages
                    avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0
                    avg_latency = sum(latencies) / len(latencies) / 10 if latencies else 0  # Scale latency
                    avg_packet_loss = sum(packet_losses) / len(packet_losses) if packet_losses else 0
                except KeyError as e:
                    logging.error(f"KeyError while extracting metrics: {e}")
                    break

                # Log results
                writer.writerow([iteration + 1, step_count + 1, avg_throughput, avg_latency, avg_packet_loss, 
                                 state["txPower"], state["channelWidth"]])

                step_count += 1

    logging.info(f"Single-band simulation results saved to {log_file}")

# Run the simulation
try:
    logging.info("Starting Single-Band Wi-Fi Simulation...")
    single_band_simulation()
finally:
    env.close()
    logging.info("Environment closed. Simulation complete.")

