import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

# --- Configuration ---
# The script will search for bag files in the folder where it is located.
BASE_FOLDER = Path('.')
POSE_TOPIC = '/vrpn_mocap/titan_alphatruck/pose'
TWIST_TOPIC = '/vrpn_mocap/titan_alphatruck/twist' # Topic for velocity data
VELOCITY_THRESHOLD = 0.05  # m/s, speed below which the robot is considered "stopped"

def process_bag(bag_path: Path):
    """
    Processes a single bag file to extract trajectory and movement duration.

    Returns:
        A tuple containing (x_positions, y_positions, duration_sec).
        Returns (None, None, None) if data cannot be processed.
    """
    print(f"--- Processing Bag: {bag_path.name} ---")
    x_positions, y_positions = [], []
    twist_data = [] # List to store (timestamp, linear_speed)

    try:
        with AnyReader([bag_path]) as reader:
            # Extract pose data for plotting
            pose_connections = [x for x in reader.connections if x.topic == POSE_TOPIC]
            if not pose_connections:
                print(f"  Warning: Pose topic '{POSE_TOPIC}' not found. Skipping path plotting.")
            else:
                for conn, ts, raw in reader.messages(connections=pose_connections):
                    msg = reader.deserialize(raw, conn.msgtype)
                    x_positions.append(msg.pose.position.x)
                    y_positions.append(msg.pose.position.y)

            # Extract twist data for time calculation
            twist_connections = [x for x in reader.connections if x.topic == TWIST_TOPIC]
            if not twist_connections:
                print(f"  Warning: Twist topic '{TWIST_TOPIC}' not found. Cannot calculate duration.")
            else:
                for conn, ts, raw in reader.messages(connections=twist_connections):
                    msg = reader.deserialize(raw, conn.msgtype)
                    speed = np.sqrt(msg.twist.linear.x**2 + msg.twist.linear.y**2)
                    twist_data.append((ts, speed))

    except Exception as e:
        print(f"  Error reading bag file {bag_path.name}: {e}")
        return None, None, None

    # Calculate movement duration
    duration_sec = None
    if twist_data:
        # Find the first time the robot started moving
        start_time = None
        for timestamp, speed in twist_data:
            if speed > VELOCITY_THRESHOLD:
                start_time = timestamp
                break

        # Find the last time the robot was moving
        end_time = None
        for timestamp, speed in reversed(twist_data):
            if speed > VELOCITY_THRESHOLD:
                end_time = timestamp
                break
        
        if start_time and end_time and end_time > start_time:
            duration_ns = end_time - start_time
            duration_sec = duration_ns / 1e9  # Convert nanoseconds to seconds
            print(f"  Movement detected. Start: {start_time}, End: {end_time}")
            print(f"  Calculated Duration: {duration_sec:.2f} seconds")
        else:
            print("  No significant movement detected based on velocity threshold.")

    return x_positions, y_positions, duration_sec

def main():
    # Find all valid ROS bag directories in the base folder recursively
    bag_paths = [p.parent for p in BASE_FOLDER.glob('**/metadata.yaml')]

    if not bag_paths:
        print(f"No ROS bag directories found in '{BASE_FOLDER.resolve()}'")
        return

    print(f"Found {len(bag_paths)} bag directories to process.")
    
    # Setup the plot
    plt.figure(figsize=(12, 10))
    
    all_durations = []

    for path in sorted(bag_paths): # Sort for consistent plotting order
        x_data, y_data, duration = process_bag(path)
        
        if duration is not None:
            all_durations.append(duration)
        
        if x_data and y_data:
            # Plot the trajectory for this run
            plt.plot(x_data, y_data, label=path.name, alpha=0.8)
            # Mark the start and end points
            plt.scatter(x_data[0], y_data[0], s=50, zorder=5) # Start
            plt.scatter(x_data[-1], y_data[-1], marker='x', s=100, zorder=5) # End

    # --- Final calculations and plotting ---
    if all_durations:
        average_time = sum(all_durations) / len(all_durations)
        print("\n" + "="*30)
        print(f"🚀 Average Movement Time: {average_time:.2f} seconds across {len(all_durations)} runs.")
        print("="*30)
    
    # Configure and show the final combined plot
    plt.title('Comparison of All Experiment Trajectories')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    output_plot_file = 'all_trajectories_comparison.png'
    plt.savefig(output_plot_file)
    print(f"\nCombined plot saved to '{output_plot_file}'")
    plt.show()


if __name__ == '__main__':
    main()
