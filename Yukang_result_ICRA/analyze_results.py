import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path

# --- Configuration ---
# Parent directory containing all result folders (e.g., Yukang_results_128)
BASE_FOLDER = Path('.') 

# ROS Topics
POSE_TOPIC = '/vrpn_mocap/titan_alphatruck/pose'
TWIST_TOPIC = '/vrpn_mocap/titan_alphatruck/twist'

# --- Analysis Parameters ---
GOAL_POSITION = np.array([4.0, -1.0]) # Target goal coordinates (x, y)
SUCCESS_RADIUS = 2.0  # meters. Trial is successful if final position is within this radius of the goal.
VELOCITY_THRESHOLD = 0.05 # m/s. Speed below which the robot is considered "stopped".

def analyze_bag(bag_path: Path):
    """
    Analyzes a single experiment run (a single bag file).

    Returns a dictionary with success status, path length, and duration.
    """
    path_points = []
    twist_data = []

    try:
        with AnyReader([bag_path]) as reader:
            # Read pose data to determine path and final position
            pose_conns = [c for c in reader.connections if c.topic == POSE_TOPIC]
            for conn, ts, raw in reader.messages(connections=pose_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                path_points.append([msg.pose.position.x, msg.pose.position.y])

            # Read twist data to determine movement time
            twist_conns = [c for c in reader.connections if c.topic == TWIST_TOPIC]
            for conn, ts, raw in reader.messages(connections=twist_conns):
                msg = reader.deserialize(raw, conn.msgtype)
                speed = np.sqrt(msg.twist.linear.x**2 + msg.twist.linear.y**2)
                twist_data.append((ts, speed))

    except Exception as e:
        print(f"  ERROR reading {bag_path.name}: {e}")
        return {'success': False, 'path_length': np.nan, 'duration': np.nan}

    if not path_points:
        return {'success': False, 'path_length': np.nan, 'duration': np.nan}

    # 1. Determine Success
    final_position = np.array(path_points[-1])
    distance_to_goal = np.linalg.norm(final_position - GOAL_POSITION)
    is_success = distance_to_goal <= SUCCESS_RADIUS

    # 2. Calculate Path Length and Duration (only if successful)
    path_length = np.nan
    duration = np.nan

    if is_success:
        # Calculate path length by summing distances between consecutive points
        path_points_np = np.array(path_points)
        path_length = np.sum(np.linalg.norm(np.diff(path_points_np, axis=0), axis=1))

        # Calculate movement duration
        if twist_data:
            start_time = next((ts for ts, speed in twist_data if speed > VELOCITY_THRESHOLD), None)
            end_time = next((ts for ts, speed in reversed(twist_data) if speed > VELOCITY_THRESHOLD), None)
            if start_time and end_time and end_time > start_time:
                duration = (end_time - start_time) / 1e9 # seconds

    return {'success': is_success, 'path_length': path_length, 'duration': duration}

def main():
    """
    Main function to traverse directories, analyze bags, and report results.
    """
    all_results = []

    # Find top-level configuration directories (e.g., Yukang_results_128)
    config_dirs = [d for d in BASE_FOLDER.iterdir() if d.is_dir() and 'Yukang_results' in d.name]

    for config_dir in sorted(config_dirs):
        print(f"\nProcessing Configuration: {config_dir.name}")
        
        # Find method directories inside (e.g., csu_logmppi_128)
        method_dirs = [d for d in config_dir.iterdir() if d.is_dir()]
        
        for method_dir in sorted(method_dirs):
            print(f"  Analyzing Method: {method_dir.name}")
            
            bag_paths = [p.parent for p in method_dir.glob('**/metadata.yaml')]
            if not bag_paths:
                continue

            trial_results = [analyze_bag(p) for p in bag_paths]
            
            # Aggregate results for this method
            num_trials = len(trial_results)
            success_count = sum(r['success'] for r in trial_results)
            
            successful_lengths = [r['path_length'] for r in trial_results if r['success']]
            successful_durations = [r['duration'] for r in trial_results if r['success']]

            avg_length = np.nanmean(successful_lengths) if successful_lengths else np.nan
            avg_duration = np.nanmean(successful_durations) if successful_durations else np.nan

            all_results.append({
                'Config': config_dir.name.split('_')[-1],
                'Method': method_dir.name,
                'Success Rate': f"{success_count}/{num_trials} ({success_count/num_trials:.0%})",
                'Avg. Path Length (m)': avg_length,
                'Avg. Time (s)': avg_duration
            })

    # --- Generate and Print Final Report Table ---
    if not all_results:
        print("No results were processed. Check folder structure and topic names.")
        return

    report_df = pd.DataFrame(all_results)
    
    # Formatting for better readability
    pd.set_option('display.precision', 2)
    pd.set_option('display.width', 120)
    
    print("\n" + "="*80)
    print(" " * 25 + "EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    print(report_df.to_string(index=False))
    print("="*80)


if __name__ == '__main__':
    main()
