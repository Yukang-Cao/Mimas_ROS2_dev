import numpy as np
import math
from numba import njit, prange

def acceleration_constraints(velocity, acceleration, v_min, v_max, a_max, v_switch):
    """
    accelerationConstraints - adjusts the acceleration based on acceleration constraints

    Inputs:
        :param acceleration - acceleration in driving direction
        :param velocity - velocity in driving direction
        :params p - longitudinal parameter structure

    Outputs:
        :return acceleration - acceleration in driving direction

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
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

def steering_constraints(steering_angle, steering_velocity, min_angle, max_angle, v_min, v_max):
    """
    steering_constraints - adjusts the steering velocity based on steering

    Inputs:
        :param steering_angle - steering angle
        :param steering_velocity - steering velocity
        :params p - steering parameter structure



    Outputs:
        :return steering_velocity - steering velocity

    Author: Matthias Althoff
    Written: 15-December-2017
    Last update: ---
    Last revision: ---
    """
    # steering limit reached?
    if (steering_angle <= min_angle and steering_velocity <= 0) or (steering_angle >= max_angle and steering_velocity >= 0):
        steering_velocity = 0
    elif steering_velocity <= v_min:
        steering_velocity = v_min
    elif steering_velocity >= v_max:
        steering_velocity = v_max

    return steering_velocity

# Define angle wrapper for yaw angle
def angle_wrap(angle):
    """
    Wrap angle to [-pi, pi].

    Parameters:
    - angle: Angle in radians.

    Returns:
    - wrapped_angle: Wrapped angle in radians.
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle

# Define angle wrapper for yaw angle between [0, 2*pi]
def angle_wrap_2pi(angle):
    """
    Wrap angle to [0, 2*pi].

    Parameters:
    - angle: Angle in radians.

    Returns:
    - wrapped_angle: Wrapped angle in radians.
    """
    wrapped_angle = angle % (2 * np.pi)
    return wrapped_angle


def convert_position_to_costmap_indices_cpu(position, map_resolution=0.05, origin=[-0.5, -2.25]):
    map_x = int(( position[0] - origin[0]) / map_resolution)
    map_y = int(( position[1] - origin[1] ) / map_resolution)
    return map_x, map_y

def convert_positions_to_costmap_indices_cpu(positions, map_resolution=0.05, origin=[-0.5, -2.25]):
    positions = np.array(positions)  # Ensure input is a NumPy array
    indices = ((positions - origin) / map_resolution).astype(int)
    map_x, map_y = indices[:, 0], indices[:, 1]  # Separate into x and y
    return map_x, map_y


# @njit(parallel=True, fastmath=True)
# def convert_trajectories_to_costmap_indices_cpu(trajs, map_resolution=0.05, origin=np.array([-0.5, -2.25],dtype=np.float32)):
#     # trajs = np.array(trajs)  # just x and y of the trajs (N,T,2)
#     trajs = np.ascontiguousarray(trajs)
#     map_x = np.zeros((trajs.shape[0], trajs.shape[1]), dtype=np.int32)
#     map_y = np.zeros((trajs.shape[0], trajs.shape[1]), dtype=np.int32)
#     for i in prange(trajs.shape[0]):
#         for t in range(trajs.shape[1]):
#             map_x[i, t] = int((trajs[i, t, 0] - origin[0]) / map_resolution)
#             map_y[i, t] = int((trajs[i, t, 1] - origin[1]) / map_resolution)
#     return map_x, map_y

@njit(parallel=True, fastmath=True, cache=True)
def convert_trajectories_to_costmap_indices_cpu(trajs, inv_res, origin_x, origin_y):
    # trajs: (N, T, 2) float32, C-contiguous
    N = trajs.shape[0]
    T = trajs.shape[1]

    map_x = np.empty((N, T), dtype=np.int32)
    map_y = np.empty((N, T), dtype=np.int32)

    total = N * T
    for idx in prange(total):
        i = idx // T
        t = idx - i * T

        x = (trajs[i, t, 0] - origin_x) * inv_res
        y = (trajs[i, t, 1] - origin_y) * inv_res

        # floor -> int32 (handles negatives correctly)
        map_x[i, t] = np.int32(np.floor(x))
        map_y[i, t] = np.int32(np.floor(y))

    return map_x, map_y

''' @brief: Returning the mean and standard deviation of the lognormal distribution, 
           given mean and variance of Normal distribution'''
def Normal2LogN(m, v):
    """ m: mean, v: variance
    Return: mu: mean, sigma: standard deviation of LN dist"""
    mu = np.exp(m + 0.5 * v)
    var = np.exp(2 * m + v) * (np.exp(v) - 1)
    sigma = np.sqrt(var)
    return mu, sigma