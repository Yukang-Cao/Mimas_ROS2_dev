import numpy as np
import math
import copy
import numba
import time
import sys 
import pickle

from scipy.ndimage import gaussian_filter, distance_transform_cdt, distance_transform_edt, binary_dilation, rotate
import matplotlib.pyplot as plt
import pdb

# from pycuda.curandom import XORWOWRandomNumberGenerator
from numba import cuda as numba_cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_normal_float32
from uge_mpc_ros2.utils.cuda_device_functions import *
from uge_mpc_ros2.utils.input_constraints import *
import os

DEFAULT_OBS_COST = 1e4
DEFAULT_DIST_WEIGHT = 1e1

class Config:
  
  """ Configurations that are typically fixed throughout execution. """
  
  def __init__(self, 
               T=5, # Horizon (s)
               dt=0.1, # Length of each step (s)
               num_control_rollouts=16384, # Number of control sequences
               num_vis_state_rollouts=16384, # Number of visualization rollouts
               seed=1, max_threads_per_block=None, max_square_block_dim=None, 
               max_blocks=None, max_rec_blocks=None, min_control_rollouts=100):
    
    self.seed = seed
    self.T = T
    self.dt = dt
    self.num_steps = int(T/dt)
    self.max_threads_per_block = max_threads_per_block # save just in case

    assert T > 0
    assert dt > 0
    assert T > dt
    assert self.num_steps > 0
    assert max_threads_per_block is not None
    assert max_square_block_dim is not None
    assert max_blocks is not None
    assert max_rec_blocks is not None
    assert min_control_rollouts is not None
    # Number of control rollouts are currently limited by the number of blocks
    self.num_control_rollouts = num_control_rollouts
    if self.num_control_rollouts > max_rec_blocks:
      self.num_control_rollouts = max_rec_blocks
      print("MPPI Config: Clip num_control_rollouts to be recommended max number of {}. (Max={})".format(
        max_rec_blocks, max_blocks))
    elif self.num_control_rollouts < min_control_rollouts:
      self.num_control_rollouts = min_control_rollouts
      print("MPPI Config: Clip num_control_rollouts to be recommended min number of {}. (Recommended max={})".format(
        min_control_rollouts, max_rec_blocks))
    
    # For visualizing state rollouts
    self.num_vis_state_rollouts = num_vis_state_rollouts
    self.num_vis_state_rollouts = min([self.num_vis_state_rollouts, self.num_control_rollouts])
    self.num_vis_state_rollouts = max([1, self.num_vis_state_rollouts])



class UAE_Numba(object):
  
  """ 


  Planner object that initializes GPU memory and runs UAE on GPU via numba.   
  CURRENT IMPLEMENATION will one calculate the cost of the each trajectory and select the one with the minimum one. 
  """

  def __init__(self, cfg):

    # Fixed configs
    self.cfg = cfg
    self.T = cfg.T
    self.dt = cfg.dt
    self.num_steps = cfg.num_steps
    self.num_control_rollouts = cfg.num_control_rollouts

    self.num_vis_state_rollouts = cfg.num_vis_state_rollouts
    self.seed = cfg.seed
    self.vehicle_length = None
    self.vehicle_width = None
    self.vehicle_wheelbase = None
    self.x0 = None
    # Basic info 
    self.max_threads_per_block = cfg.max_threads_per_block

    # Initialize reuseable device variables
    self.u_cur_d = None
    self.u_prev_d = None
    self.costs_d = None
    self.feasible_mask_d = None
    self.num_feasible_d = np.float32(0.0)
    self.weights_d = None
    self.rng_states_d = None
    self.state_rollout_batch_d = None # For visualization only. Otherwise, inefficient

    # Other task specific params
    self.device_var_initialized = False
    # self.generator = XORWOWRandomNumberGenerator()

    self.max_T = 1024
    # Task specific params
    self.costmap_meta = None
    self.costmap_mins = None
    # local costmap size and resolution
    # self.local_costmap_size = 400 # 400 * 0.05 = 20m
    self.local_costmap_resolution = 0.05
    self.local_costmap_range = 6.0
    self.local_costmap_map = None

    self.reset()

  def reset(self):
    # Other task specific params
    self.u_seq0 = np.zeros((self.num_steps, 2), dtype=np.float32)
    # self.u_seq0[:,0] = 1.0 # Linear velocity
    self.params = None
    self.params_set = False
    self.costmap_loaded = False
    self.local_costmap_loaded = False
    self.u_prev_d = None
    
    # Initialize all fixed-size device variables ahead of time. (Do not change in the lifetime of MPPI object)

    self.init_device_vars_before_solving()

  def load_trajectories(self, trajectories):
    self.trajectories = copy.deepcopy(trajectories)
    # Generate zeros_like trajectories for transformed trajectories
    # self.transformed_trajectories = np.zeros_like(trajectories)

    self.trajectories_d = numba_cuda.to_device(self.trajectories.astype(np.float32))
    self.original_trajectories_d = numba_cuda.to_device(self.trajectories.astype(np.float32))

  def init_device_vars_before_solving(self):

    if not self.device_var_initialized:
      t0 = time.time()
      # Useq
      self.u_cur_d = numba_cuda.to_device(self.u_seq0) 
      self.u_prev_d = numba_cuda.to_device(self.u_seq0)

      self.costs_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      # self.feasible_mask_d = numba_cuda.device_array((self.num_control_rollouts, self.num_steps+1), dtype=np.float32)
      # # add full ones to the feasible mask
      # self.feasible_mask_d[0:self.num_control_rollouts, 0:self.num_steps+1] = 1.0
      # # self.num_feasible_d = np.float32(0.0)
      self.weights_d = numba_cuda.device_array((self.num_control_rollouts), dtype=np.float32)
      # self.local_costmap_d = numba_cuda.device_array((self.local_costmap_size, self.local_costmap_size), dtype=np.float32)         
      self.debug_d = numba_cuda.device_array((self.num_control_rollouts, self.num_steps+1, 4), dtype=np.float32)
      self.device_var_initialized = True
      print(" UAE planner has initialized GPU memory after {} s".format(time.time()-t0))

  def setup(self, params):
    # These tend to change (e.g., current robot position, the map) after each step
    self.set_params(params)

  def set_params(self, params):
    self.params = copy.deepcopy(params)
    self.x0 = self.params['x0']
    self.vehicle_length = self.params['vehicle_length']
    self.vehicle_width = self.params['vehicle_width']
    self.vehicle_wheelbase = self.params['vehicle_wheelbase']
    self.params_set = True
    # self.load_costmap(self.params['costmap'], self.params['map_mins'])

  def load_costmap(self, costmap):
    # COSTMAP
    costmap_full = copy.deepcopy(costmap.astype(np.float32))
    # Padding the costmap to avoid index out of bounds
    self.costmap_full_padded = costmap_full
    self.costmap_loaded = True
    # self.costmap_mins = map_mins
    # print(f"costmap_mins: {self.costmap_mins}")


  def check_solve_conditions(self):
    if not self.params_set:
      print("MPPI parameters are not set. Cannot solve")
      return False
    if not self.device_var_initialized:
      print("Device variables not initialized. Cannot solve.")
      return False
    if not self.costmap_loaded:
      print("Local costmap not loaded. Cannot solve.")
      return False
    return True

  def solve_uae(self, trajectories):
    """Entry point for different algoritims"""
    
    if not self.check_solve_conditions():
      print("UAE solve condition not met. Cannot solve. Return")
      return
    # TODO: HARD CODED NUMBER OF TRAJECTORIES !!!!
    return self.get_rollout_cost_uae(trajectories, num_trajectories=trajectories.shape[0])
  
  def solve_mppi(self):
    """Entry point for different algoritims"""
    
    if not self.check_solve_conditions():
      print("MPPI solve condition not met. Cannot solve. Return")
      return
    
    return self.get_rollout_cost_uae()
  
  def convert_position_to_costmap_indices(self, position, map_mins, resolution=0.05): 
    map_resolution = resolution
    origin = map_mins
    # origin = [-2.25, -0.5]
    # origin = [-0.5-3,-2.25-3]
    map_x = int((position[0] - origin[0]) / map_resolution)
    map_y = int((position[1] - origin[1] ) / map_resolution)
    return map_x, map_y
  
  def _disk(self,radius_cells: int) -> np.ndarray:
    """Binary circular footprint with given integer radius in cells."""
    if radius_cells <= 0:
        return np.ones((1, 1), dtype=bool)
    r = int(radius_cells)
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (xx*xx + yy*yy) <= (r*r)

  def generate_linear_decay_costmap(self,
    grid: np.ndarray,
    inflation_radius: float = 3.0,
    max_inflation_value: float = 1.0,
    *,
    # NEW: pre-inflate obstacles before decay
    pre_inflate_cells: int = 0,
    # Optional metric interface (set map_resolution>0 and use *_meters)
    map_resolution: float = 0.0,            # meters per cell; 0 -> ignore metric args
    pre_inflate_meters: float = 0.0,        # used if map_resolution>0
    decay_radius_meters: float = 0.0,       # used if map_resolution>0 (overrides inflation_radius)
    # Handling unknowns
    unknown_value: int | float | None = None,  # e.g., -1 for ROS costmaps; None to ignore
    treat_unknown_as_obstacle: bool = False
    ) -> np.ndarray:
    """
    Build a costmap by (1) dilating obstacles and (2) applying linear decay
    from the dilated obstacle boundary.

    Args:
        grid: 2D occupancy/cost grid. Obstacles are grid>0 by default.
        inflation_radius: (cells) linear decay reach if metric args not used.
        max_inflation_value: max cost contributed by decay (0..1 typical).
        pre_inflate_cells: (cells) direct dilation before decay.
        map_resolution: meters/cell. If >0, you can specify metric radii.
        pre_inflate_meters: meters for direct dilation (overrides pre_inflate_cells).
        decay_radius_meters: meters for decay reach (overrides inflation_radius).
        unknown_value: value representing unknown (e.g., -1). Leave None to ignore.
        treat_unknown_as_obstacle: if True, unknowns are dilated like obstacles.

    Returns:
        Float32 costmap with direct inflation + linear decay combined with original grid.
    """
    grid = np.asarray(grid)
    H, W = grid.shape

    # --- Identify obstacles (and optionally unknowns) ---
    obstacle_map = grid > 0
    if unknown_value is not None and treat_unknown_as_obstacle:
        obstacle_map = obstacle_map | (grid == unknown_value)

    # --- Determine radii (cells) ---
    if map_resolution and map_resolution > 0.0:
        # Metric interface in meters
        pre_r_cells = int(np.floor(pre_inflate_meters / map_resolution + 1e-6))
        decay_r_cells = max(1, int(np.floor(decay_radius_meters / map_resolution + 1e-6))) \
                        if decay_radius_meters > 0 else int(max(1, round(inflation_radius)))
    else:
        pre_r_cells = int(max(0, pre_inflate_cells))
        decay_r_cells = int(max(1, round(inflation_radius)))

    # --- Step 1: direct dilation (pre-inflation) ---
    if pre_r_cells > 0:
        footprint = self._disk(pre_r_cells)
        inflated_obstacles = binary_dilation(obstacle_map, structure=footprint)
    else:
        inflated_obstacles = obstacle_map

    # --- Step 2: distance transform from pre-inflated obstacles ---
    # distance is 0 inside obstacles; grows into free space
    distance_map = distance_transform_edt(~inflated_obstacles)

    # Linear decay within decay_r_cells:
    # cost = max_value * clip((R - d)/R, 0, 1)
    linear_decay_costmap = np.clip(
        (decay_r_cells - distance_map) / max(decay_r_cells, 1),
        0.0, 1.0
    ) * float(max_inflation_value)

    # --- Combine with original grid (keep existing obstacle cost) ---
    costmap = np.maximum(grid.astype(np.float32), linear_decay_costmap.astype(np.float32))

    # Preserve unknowns if not treating them as obstacles and they exist
    if unknown_value is not None and not treat_unknown_as_obstacle:
        mask_unknown = (grid == unknown_value)
        costmap[mask_unknown] = unknown_value

    return costmap

  # def generate_linear_decay_costmap(self, grid, inflation_radius=3, max_inflation_value=1.0):
  #     """
  #     Generate an inflated costmap based on linear decay from obstacles.

  #     Args:
  #         grid (np.ndarray): Original grid map with obstacles.
  #         inflation_radius (float): Radius for inflation in grid cells.
  #         max_inflation_value (float): Maximum cost value for inflated cells.

  #     Returns:
  #         np.ndarray: Inflated costmap.
  #     """
  #     # Generate binary obstacle map
  #     obstacle_map = grid > 0

  #     # Compute Euclidean distance transform
  #     distance_map = distance_transform_edt(~obstacle_map)

  #     #Before applying the linear decay, inflate the
      
  #     # Apply linear decay
  #     linear_decay_costmap = np.clip((inflation_radius - distance_map) / inflation_radius, 0, 1) * max_inflation_value

  #     # Combine the linear decay costmap with the original grid
  #     costmap = np.maximum(grid, linear_decay_costmap).astype(np.float32)

  #     return costmap
  
  def move_uae_task_vars_to_device(self, trajectories):

    R_Matrix = np.array([[math.cos(-self.params['x0'][2]), -math.sin(-self.params['x0'][2])], [math.sin(-self.params['x0'][2]), math.cos(-self.params['x0'][2])]])
    x_goal_r = np.dot(R_Matrix, (self.params['xgoal'][:2].astype(np.float32)- self.params['x0'][:2].astype(np.float32))).T 
    xgoal_d = numba_cuda.to_device(np.concatenate((x_goal_r, self.params['xgoal'][2:].astype(np.float32)), axis=0))
    # print(f"xgoal_d: {xgoal_d.copy_to_host()}")
    # xgoal_d = numba_cuda.to_device(self.params['xgoal'].astype(np.float32))
    x0_d = numba_cuda.to_device(self.params['x0'].astype(np.float32))
    goal_tolerance_d = np.float32(self.params['goal_tolerance'])

    vehicle_length_d = np.float32(self.vehicle_length)
    vehicle_width_d = np.float32(self.vehicle_width)
    vehicle_wheelbase_d = np.float32(self.vehicle_wheelbase)


    #USEQ UPDATE
    lambda_weight_d = np.float32(self.params['lambda_weight'])
    vrange_d = np.array(self.params['vrange'], dtype=np.float32)
    wrange_d = np.array(self.params['wrange'], dtype=np.float32)
    # read

    # # COSTMAP
    # make the local_costmap contigous
    local_costmap_edt = copy.deepcopy(self.costmap_full_padded)
    local_costmap_edt = np.ascontiguousarray(local_costmap_edt)
    self.local_costmap_map = local_costmap_edt
    local_costmap_d = numba_cuda.to_device(local_costmap_edt)

    obs_cost_d = np.float32(DEFAULT_OBS_COST if 'obs_penalty' not in self.params 
                                  else self.params['obs_penalty'])
    local_costmap_mins_d = numba_cuda.to_device(np.array([-self.local_costmap_range, -self.local_costmap_range]).astype(np.float32))
    # set the trajectories to the trajectories_d on the device
    # self.trajectories_d = numba_cuda.to_device(trajectories.astype(np.float32)) # no need to return this
    # max_local_cost_d = np.float32(25.0)
    max_local_cost_d = np.float32(81.0)
    
    return xgoal_d, x0_d, goal_tolerance_d, \
            vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
            lambda_weight_d, vrange_d, wrange_d, \
            obs_cost_d, local_costmap_d, max_local_cost_d, local_costmap_mins_d

  def get_rollout_cost_uae(self, trajectories, num_trajectories=None):
   
    '''
    Calculate the cost of the each trajectories and find the trajectory with min cost. and return that trajectory to the host
    '''

    xgoal_d, x0_d, goal_tolerance_d, \
      vehicle_length_d, vehicle_width_d, vehicle_wheelbase_d, \
      lambda_weight_d, vrange_d, wrange_d, \
        obs_cost_d, local_costmap_d, max_local_cost_d, local_costmap_mins_d = self.move_uae_task_vars_to_device(trajectories)
    # print(f"x0_d: {x0_d.copy_to_host()}")
    # print(f"xgoal_d: {xgoal_d.copy_to_host()}")
    # Weight for distance cost
    dist_weight_d = DEFAULT_DIST_WEIGHT if 'dist_weight' not in self.params else self.params['dist_weight']

    if num_trajectories is None:
      num_trajectories = self.num_control_rollouts

    num_trajectories_d = np.float32(num_trajectories)

    # print(f"num_trajectories_d: {num_trajectories_d}")
    #Transform the trajectories to the current state
    threadperblock = (16,16)
    # print(self.num_control_rollouts)
    blockpergrid_x = (self.num_control_rollouts + threadperblock[0] - 1) // threadperblock[0]
    blockpergrid_y = (self.num_steps + threadperblock[1] - 1) // threadperblock[1]
    blockpergrid = (blockpergrid_x, blockpergrid_y)
    self.transform_trajectories[blockpergrid, threadperblock](x0_d, self.original_trajectories_d, self.trajectories_d)

    

    #--------------------------------
    self.rollouts_cost_numba[num_trajectories, 1](
      self.trajectories_d,
      self.costs_d,
      goal_tolerance_d,
      xgoal_d,
      x0_d,
      obs_cost_d,
      local_costmap_d,
      max_local_cost_d,
      dist_weight_d,
      num_trajectories_d,
      local_costmap_mins_d   
    )

    # # get the cost of the trajectories that are feasible to the host
    cost_arr = self.costs_d.copy_to_host()
    # round the cost_arr to 2 decimal places
    cost_arr_rounded = np.round(cost_arr, 3)  

    min_cost_index = np.argmin(cost_arr[:self.trajectories_d.shape[0]])

    return min_cost_index, cost_arr[:self.trajectories_d.shape[0]]

  def get_vehicle_boundary_points_p(self, x_curr, vehicle_length, vehicle_width):
    x_center, y_center, theta = x_curr
    # Half dimensions
    half_length = vehicle_length / 2
    half_width = vehicle_width / 2

    # Define the relative positions of the corners
    corners = np.array([
        [half_length, half_width],     # Front left
        [half_length, -half_width],    # Front right
        [-half_length, -half_width],   # Rear right
        [-half_length, half_width]     # Rear left
    ])

    # Compute the rotation matrix based on heading angle (theta)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Rotate corners by the heading angle and translate to world coordinates
    world_corners = rotation_matrix @ corners.T 
    world_corners = world_corners.T + np.array([x_center, y_center])
    # Add first point to the end for visualization
    world_corners = np.vstack([world_corners, world_corners[0]])
    return world_corners
  
  def get_state_rollout(self, x_curr, trajectories):

    # First translate the point
    transformed_trajectories = copy.deepcopy(trajectories)

    # Then rotate the point
    # Rotation matrix
    R = np.array([[math.cos(x_curr[2]), -math.sin(x_curr[2])], [math.sin(x_curr[2]), math.cos(x_curr[2])]])
    for i in range(transformed_trajectories.shape[0]):
      transformed_trajectories[i,:,:2] = np.dot(R, transformed_trajectories[i,:,:2].T).T      
      # add the theta
      transformed_trajectories[i,:,2] += x_curr[2]
      # Normalize theta between 0 and 2pi
      # transformed_trajectories[i,:,2] = math.fmod(transformed_trajectories[i,:,2], 2*np.pi)
    for i in range(transformed_trajectories.shape[0]):
      transformed_trajectories[i,:,0] += x_curr[0]
      transformed_trajectories[i,:,1] += x_curr[1]
    return transformed_trajectories
    
  def shift_and_update(self, x_next, useq):

    self.x0 = self.params["x0"] = x_next.copy()

    u_cur_shifted = useq.copy()
    u_cur_shifted[:-1] = u_cur_shifted[1:]
    # u_cur_shifted[-1][1] = 0.0
    self.u_cur_d = numba_cuda.to_device(u_cur_shifted.astype(np.float32))
      
  """GPU kernels from here on"""


  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def rollouts_cost_numba(trajectories_d,
    costs_d,
    goal_tolerance_d,
    xgoal_d,
    x0_d,
    obs_cost_d,
    costmap_d,
    max_local_cost_d,
    dist_weight_d,
    num_trajectories_d,
    map_mins_d
  ):
    """
    There should only be one thread running in each block, where each block handles a single sampled trajecotry calulation.
    """
    
    # Get block id and thread id
    bid = numba_cuda.blockIdx.x   # index of block
    tid = numba_cuda.threadIdx.x  # index of thread within a block


    if bid < num_trajectories_d:
      # Initialize the cost for the trajectory
      costs_d[bid] = 0.0
      goal_reached = False
      isCollided = numba_cuda.local.array(1, dtype=numba.float32)
      isCollided[0] = 0.0
      goal_tolerance_d2 = goal_tolerance_d*goal_tolerance_d
      dist_to_goal2 = prev_dist_to_goal2 = 1e9 # initialize to a large value

      # Allocate space for vehicle boundary points (4)
      x_curr = numba_cuda.local.array(3, numba.float32) # Kinematic car model states x,y,theta
      x_curr_grid_d = numba_cuda.local.array((2), dtype=np.int32)
      
      # Loop through each state in the trajectory
      num_steps = trajectories_d.shape[1]
      
      for step in range(num_steps):
          # Extract current state (x, y, theta)
          for i in range(3):
            x_curr[i] = trajectories_d[bid, step, i]
          # check for collision
          if isCollided[0] == 0.0:
            # Get current state costmap indices
            convert_position_to_costmap_indices_TO_gpu(x_curr[0],x_curr[1], map_mins_d[0], map_mins_d[1], 0.05, x_curr_grid_d)

            # Check for collision 
            # if costmap incides are out of bounds, then act as a free space
            if x_curr_grid_d[0] < 0 or x_curr_grid_d[0] >= costmap_d.shape[1] or x_curr_grid_d[1] < 0 or x_curr_grid_d[1] >= costmap_d.shape[0]:
              # costs_d[bid] += 1 * obs_cost_d
              costs_d[bid] += 0.0       
            else:
              if check_state_collision_gpu(costmap_d, x_curr_grid_d) == 1.0:
                isCollided[0] = 1.0
            
              costs_d[bid] += (calculate_localcostmap_cost(costmap_d, x_curr_grid_d) / max_local_cost_d) * obs_cost_d # footprint cost
              # costs_d[bid] += (calculate_localcostmap_cost(costmap_d, x_curr_grid_d) / 1.0) * obs_cost_d

            # Compute distance to goal
            dist_to_goal2 = (((xgoal_d[0]-x_curr[0])**2 + (xgoal_d[1]-x_curr[1])**2))**0.5
            costs_d[bid] += stage_cost(dist_to_goal2, dist_weight_d) # for trajectory optimization exp
            if dist_to_goal2  <= goal_tolerance_d:
              goal_reached = True
              break

            prev_dist_to_goal2 = dist_to_goal2
          else:
            costs_d[bid] += 1 * obs_cost_d
            # distans
            costs_d[bid] += prev_dist_to_goal2 
      # Accumulate terminal cost
      costs_d[bid] += term_cost_additional_weight(dist_to_goal2, goal_reached, 30.0)

    else: # if bid >= num_trajectories_d
        costs_d[bid] = 1e9

  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def update_useq_numba(
        lambda_weight_d,
        costs_d,
        trajectories_d,
        weights_d,
        vrange_d,
        wrange_d,
        u_cur_d,
        u_prev_d):
    """
    GPU kernel that updates the optimal control sequence based on previously evaluated cost values.
    Assume that the function is invoked as update_useq_numba[1, NUM_THREADS], with one block and multiple threads.
    """

    tid = numba_cuda.threadIdx.x
    num_threads = numba_cuda.blockDim.x
    numel = len(costs_d)
    gap = int(math.ceil(numel / num_threads))

    # Find the minimum value via reduction
    starti = min(tid*gap, numel)
    endi = min(starti+gap, numel)
    if starti<numel:
      weights_d[starti] = costs_d[starti]
    for i in range(starti, endi):
      weights_d[starti] = min(weights_d[starti], costs_d[i])
    numba_cuda.syncthreads()

    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        weights_d[starti] = min(weights_d[starti], weights_d[starti + s])
      s *= 2
      numba_cuda.syncthreads()

    beta = weights_d[0]
    
    # Compute weight
    for i in range(starti, endi):
      weights_d[i] = math.exp(-1./lambda_weight_d*(costs_d[i]-beta))
    numba_cuda.syncthreads()

    # Normalize
    # Reuse costs_d array
    for i in range(starti, endi):
      costs_d[i] = weights_d[i]
    numba_cuda.syncthreads()
    for i in range(starti+1, endi):
      costs_d[starti] += costs_d[i]
    numba_cuda.syncthreads()
    s = gap
    while s < numel:
      if (starti % (2 * s) == 0) and ((starti + s) < numel):
        # Stride by `s` and add
        costs_d[starti] += costs_d[starti + s]
      s *= 2
      numba_cuda.syncthreads()

    for i in range(starti, endi):
      weights_d[i] /= costs_d[0]
    numba_cuda.syncthreads()
    
    # update the u_cur_d
    timesteps = len(u_cur_d)
    for t in range(timesteps):
      for i in range(starti, endi):
        # numba_cuda.atomic.add(u_cur_d, (t, 0), weights_d[i]*trajectories_d[i, t, 0])
        numba_cuda.atomic.add(u_cur_d, (t, 1), weights_d[i]*(trajectories_d[i, t, 3]-u_prev_d[t, 1]))
    numba_cuda.syncthreads()

    # Blocks crop the control together
    tgap = int(math.ceil(timesteps / num_threads))
    starti = min(tid*tgap, timesteps)
    endi = min(starti+tgap, timesteps)
    for ti in range(starti, endi):
      # u_cur_d[ti, 0] = max(vrange_d[0], min(vrange_d[1], u_cur_d[ti, 0]))
      u_cur_d[ti, 1] = max(wrange_d[0], min(wrange_d[1], u_cur_d[ti, 1]))
  
  @staticmethod
  @numba_cuda.jit(fastmath=True)
  def transform_trajectories(
    x_curr,
    trajectories,
    transformed_trajectories
  ):
    """
    Transform the trajectories to the current state.
    """
    i, j = numba_cuda.grid(2)

    if i < trajectories.shape[0] and j < trajectories.shape[1]:
       # precompute sin and cos
      cos_theta = math.cos(x_curr[2])
      sin_theta = math.sin(x_curr[2])
      x_curr_x = x_curr[0]
      x_curr_y = x_curr[1]
      x_curr_theta = x_curr[2]

      # load the trajectory
      x = trajectories[i, j, 0] - x_curr_x
      y = trajectories[i, j, 1] - x_curr_y
      theta = trajectories[i, j, 2]

      # Rotate the trajectory
      transformed_trajectories[i, j, 0] = x * cos_theta + y * sin_theta 
      transformed_trajectories[i, j, 1] = -x * sin_theta + y * cos_theta 
      transformed_trajectories[i, j, 2] = theta + x_curr_theta
      # normalize theta between 0 and 2pi
      transformed_trajectories[i, j, 2] = math.fmod(transformed_trajectories[i, j, 2], 2*np.pi)
  
  