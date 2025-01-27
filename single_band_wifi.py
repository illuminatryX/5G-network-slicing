#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import random
import time
from collections import deque

import pandas as pd

# Setup logging for better traceability
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Command-line argument parsing
def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Single-Band Wi-Fi Simulation")
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations, Default: 10')
    parser.add_argument('--log', type=str, default='single_band_log.csv', help='Log file name, Default: single_band_log.csv')
    return parser.parse_args()

# Parameters for Single-Band Simulation
def single_band_params():
    """Defines single-band Wi-Fi parameters."""
    return {
        'channelNumber': 1,  # Single band channel
        'channelWidth': 20,  # Reduced width
        'gi': 800,           # Guard interval
        'mcs': 4,            # Moderate MCS
        'txPower': 18        # Slightly reduced power
    }

# Generate Simulation Metrics
def generate_metrics(iterations, params):
    """Simulates metrics for single-band Wi-Fi."""
    metrics = []
    for i in range(1, iterations + 1):
        step = random.randint(100, 500) * i  # Simulate incremental steps
        reward = random.randint(800, 1500)  # Simulate rewards
        latency = random.randint(50, 300) * i  # Simulate increasing latency
        tx_packets = random.randint(1000, 5000) * i  # Simulate transmitted packets
        rx_packets = tx_packets - random.randint(0, 500)  # Some packets lost
        metrics.append({
            'iteration': i,
            'step': step,
            'reward': reward,
            'latency': latency,
            'tx_packets': tx_packets,
            'rx_packets': rx_packets,
            **params
        })
    return metrics

def main():
    args = parse_arguments()
    iterations = args.iterations
    log_file = args.log

    # Define single-band Wi-Fi parameters
    params = single_band_params()

    # Generate simulation results
    logging.info("Generating single-band Wi-Fi simulation results...")
    results = generate_metrics(iterations, params)

    # Save results to a CSV file
    logging.info(f"Saving results to {log_file}")
    df = pd.DataFrame(results)
    df.to_csv(log_file, index=False)
    logging.info(f"Simulation complete. Results saved to {log_file}.")

if __name__ == "__main__":
    main()

