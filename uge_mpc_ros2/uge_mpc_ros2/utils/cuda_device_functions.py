import numba.cuda
import numpy as np
import math
import copy
import numba
import time
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
import matplotlib.pyplot as plt
import pdb



# Stage costs (device function)
@cuda.jit('float32(float32, float32)', device=True, inline=True)
def stage_cost(dist2, dist_weight):
    return dist_weight*dist2 # squared term makes the robot move faster

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean)', device=True, inline=True)
def term_cost(dist2, goal_reached):
    return (1-np.float32(goal_reached))*dist2

# Terminal costs (device function)
@cuda.jit('float32(float32, boolean, float32)', device=True, inline=True)
def term_cost_additional_weight(dist2, goal_reached, dist_weight):
    return ((1-np.float32(goal_reached))*dist2)*dist_weight


@cuda.jit('void(float32[:], float32, float32, float32[:,:])', device=True, inline=True)
def get_vehicle_boundary_points(x_curr, vehicle_length, vehicle_width, vehicle_boundary_points):
    x_center = x_curr[0]
    y_center = x_curr[1]
    theta = x_curr[4] # heading angle
    # epsilon = 0.1 outer bound for the car
    half_length = (vehicle_length / 2) + 0.1
    half_width = (vehicle_width / 2) + 0.1

    # Define relative corner positions
    corners = cuda.local.array((6, 2), dtype=np.float32)
    corners[0, 0], corners[0, 1] = half_length, half_width    # Front left
    corners[1, 0], corners[1, 1] = half_length, -half_width   # Front right
    corners[2, 0], corners[2, 1] = 0.0, -half_width   # center right
    corners[3, 0], corners[3, 1] = -half_length, -half_width  # Rear right
    corners[4, 0], corners[4, 1] = -half_length, half_width   # Rear left
    corners[5, 0], corners[5, 1] = 0.0, half_width   # Center left
    '''
        corners = np.array([
        [half_length, half_width],     # Front left
        [half_length, -half_width],    # Front right
        [0, -half_width],               # center right,   
        [-half_length, -half_width],   # Rear right
        [-half_length, half_width],    # Rear left,
        [0, half_width]    # center left
    ])'''
    # Precompute trigonometric functions for rotation
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # Rotate each corner using the vehicle's heading, then translate to global coordinates
    for i in range(6):
        rotated_x = corners[i, 0] * cos_theta - corners[i, 1] * sin_theta
        rotated_y = corners[i, 0] * sin_theta + corners[i, 1] * cos_theta

        vehicle_boundary_points[i, 0] = rotated_x + x_center
        vehicle_boundary_points[i, 1] = rotated_y + y_center


#Convert the vehicle boundary points to the grid coordinates
@cuda.jit('void(float32[:,:], float32, float32, float32, float32, float32[:,:])', device=True, inline=True)
def get_vehicle_boundary_points_grid(vehicle_boundary_points, x_min, y_min, grid_resolution,scale, vehicle_boundary_points_grid):
    for i in range(4):
        x = vehicle_boundary_points[i, 0]
        y = vehicle_boundary_points[i, 1]

        x_grid = int(((scale * x - x_min)) / grid_resolution)
        y_grid = int(((scale * y - y_min)) / grid_resolution)

        vehicle_boundary_points_grid[i, 0] = x_grid
        vehicle_boundary_points_grid[i, 1] = y_grid

#Convert the vehicle current position to the grid coordinates
@cuda.jit('void(float32, float32, float32, float32, float32, int32[:])', device=True, inline=True)
def convert_position_to_costmap_indices_gpu(x_curr_x, x_curr_y, x_min, y_min, grid_resolution, x_curr_grid):

    x = x_curr_x
    y = x_curr_y

    x_grid = numba.int32((x - x_min) / grid_resolution)
    y_grid = numba.int32((y - y_min) / grid_resolution)

    x_curr_grid[0] = x_grid
    x_curr_grid[1] = y_grid

#Convert the vehicle current position to the grid coordinates
@cuda.jit('void(float32, float32, float32, float32, float32, int32[:])', device=True, inline=True)
def convert_position_to_costmap_indices_TO_gpu(x_curr_x, x_curr_y, x_min, y_min, grid_resolution, x_curr_grid):

    x = x_curr_x
    y = x_curr_y

    x_grid = numba.int32((x - x_min) / grid_resolution)
    y_grid = numba.int32((y - y_min) / grid_resolution)

    x_curr_grid[0] = x_grid
    x_curr_grid[1] = y_grid



@cuda.jit('float32(float32[:,:], float32, float32, float32[:,:])', device=True, inline=True)
def calculate_obstacle_cost(vehicle_boundary_points_grid, obstacle_weight, max_cost, costmap):
    cost = 0.0
    for i in range(4):
        corner_x = vehicle_boundary_points_grid[i, 0]
        corner_y = vehicle_boundary_points_grid[i, 1]
        cost += obstacle_weight*(costmap[int(corner_y), int(corner_x)])
    return cost

@cuda.jit('float32(float32[:,:], int32[:])', device=True, inline=True)
def calculate_localcostmap_cost(costmap, x_curr_grid):
    cost = 0.0
    for i in range(9):
        for j in range(9):
            cost += costmap[x_curr_grid[1]-1+i, x_curr_grid[0]-1+j]
    # cost += costmap[x_curr_grid[1], x_curr_grid[0]] # single point cost
    return cost

@cuda.jit('float32(float32[:,:], int32[:])', device=True, inline=True)
def check_state_collision_gpu(costmap, x_curr_grid):
    for i in range(9):
        for j in range(9):
            if costmap[x_curr_grid[1]-1+i, x_curr_grid[0]-1+j] == 1:
                return 1.0 # collision
    return 0.0 # no collision



# @cuda.jit('float32(float32[:,:], float32[:,:], float32[:], float32)', device=True, inline=True)
# def calculate_obstacle_cost(vehicle_boundary_points, obs_pos_d, obs_r_d, obs_cost_d):
#     num_obs = len(obs_pos_d)
#     cost = 0.0

#     for obs_i in range(num_obs):
#         obs_x = obs_pos_d[obs_i, 0]
#         obs_y = obs_pos_d[obs_i, 1]
#         obs_radius_sq = obs_r_d[obs_i]

#         for i in range(4):
#             corner_x = vehicle_boundary_points[i, 0]
#             corner_y = vehicle_boundary_points[i, 1]
#             dist_sq = (corner_x - obs_x) ** 2 + (corner_y - obs_y) ** 2 - obs_radius_sq**2
#             cost += (1-numba.float32(dist_sq>0))*obs_cost_d

#     return cost


@cuda.jit('float32(float32, float32, float32, float32,float32, float32)', device=True, inline=True)
def steering_contrainsts_cuda(steering_angle, steering_velocity, min_angle, max_angle, v_min, v_max):

    # Check if steering limit is reached
    if (steering_angle <= min_angle and steering_velocity <= 0) or (steering_angle >= max_angle and steering_velocity >= 0):
        steering_velocity = 0
    elif steering_velocity <= v_min:
        steering_velocity = v_min
    elif steering_velocity >= v_max:
        steering_velocity = v_max

    return steering_velocity



@cuda.jit('float32(float32, float32, float32, float32, float32, float32)', device=True, inline=True)
def acceleration_contrainsts_cuda(velocity, acceleration, v_switch, a_max, v_min, v_max):

    # positive acceleration limit
    if velocity > v_switch:
        posLimit = a_max * v_switch / velocity
    else:
        posLimit = a_max

    # acceleration limit reached?
    if (velocity <= v_min and acceleration <= 0) or (velocity >= v_max and acceleration >= 0):
        acceleration = 0
    elif acceleration <= -a_max:
        acceleration = -a_max
    elif acceleration >= posLimit:
        acceleration = posLimit

    return acceleration





# Define angle wrapper in cuda device function for yaw angle bbetween -pi and pi
@cuda.jit('float32(float32)', device=True, inline=True)
def angle_wrap_cuda(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Define angle wrapper in cuda device function for yaw angle between 0 and 2pi
@cuda.jit('float32(float32)', device=True, inline=True)
def angle_wrap_2pi_cuda(angle):
    return angle % (2 * np.pi)

# ---- helpers ---------------------------------------------------------------

@cuda.jit(device=True, inline=True)
def wrap_to_pi(a):
    # wrap to (-pi, pi]
    two_pi = 2.0 * math.pi
    while a <= -math.pi:
        a += two_pi
    while a > math.pi:
        a -= two_pi
    return a

@cuda.jit(device=True, inline=True)
def angle_diff_abs(a, b):
    return abs(wrap_to_pi(a - b))

@cuda.jit(device=True)
def action_smoothness_upto_L(actions_2d, L):
    """
    actions_2d: (T-1, 2) view of actions for a single trajectory (float32)
    L: number of *action* timesteps to include (<= T-1). If L<=1 -> 0.
    Returns sum_{t=1..L-1} ||u_t - u_{t-1}||^2
    """
    if L <= 1:
        return 0.0
    acc = 0.0
    for t in range(1, L):
        du0 = actions_2d[t, 0] - actions_2d[t-1, 0]
        du1 = actions_2d[t, 1] - actions_2d[t-1, 1]
        acc += du0 * du0 + du1 * du1
    return acc
