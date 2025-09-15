import matplotlib.pyplot as plt
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path

# --- Configuration ---
BAG_FILE_PATH = Path('experiment_run_1')
POSE_TOPIC = '/vrpn_mocap/titan_alphatruck/pose'
OUTPUT_CSV_FILE = 'trajectory.csv'
OUTPUT_PLOT_FILE = 'trajectory.png'

def main():
    x_positions = []
    y_positions = []

    if not BAG_FILE_PATH.exists():
        print(f"Error: Bag directory not found at '{BAG_FILE_PATH}'")
        return

    try:
        # Use AnyReader to open the bag file directory
        with AnyReader([BAG_FILE_PATH]) as reader:
            # THIS IS THE KEY CHANGE:
            # Directly filter for the topic you want in the messages() generator.
            # This avoids accessing attributes that may not exist.
            connections = [x for x in reader.connections if x.topic == POSE_TOPIC]
            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                x_positions.append(msg.pose.position.x)
                y_positions.append(msg.pose.position.y)

    except Exception as e:
        print(f"An error occurred while reading the bag file: {e}")
        return

    if not x_positions:
        print(f"Error: No messages were extracted from topic '{POSE_TOPIC}'.")
        print("Please check if the topic name is correct and if the bag file contains data for it.")
        return
        
    print(f"Successfully extracted {len(x_positions)} data points.")

    # --- Save to CSV ---
    df = pd.DataFrame({'x': x_positions, 'y': y_positions})
    df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Trajectory data saved to {OUTPUT_CSV_FILE}")

    # --- Plotting ---
    plt.figure(figsize=(10, 8))
    plt.plot(x_positions, y_positions, label='Robot Path')
    plt.scatter(x_positions[0], y_positions[0], color='green', s=100, zorder=5, label='Start')
    plt.scatter(x_positions[-1], y_positions[-1], color='red', s=100, zorder=5, label='End')
    plt.title('Robot Trajectory from Experiment')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"Plot saved to {OUTPUT_PLOT_FILE}")
    plt.show()

if __name__ == '__main__':
    main()
