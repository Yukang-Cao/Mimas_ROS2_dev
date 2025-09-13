# lib/torch_planner_base.py
"""
Intermediate base class for PyTorch-based trajectory planners.
Centralizes dynamics, cost evaluation, and perception handling to ensure consistency
across different sampling strategies (C-Uniform, MPPI, etc.).
"""

import numpy as np
import torch
import os
import time
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

from .base_controller import BaseController, PlannerInput

from .utils.dynamics import ( # import the centralized dynamics functions using relative import
    cuda_dynamics_KS_3d_scalar_v_batched as dynamics_scalar_torch,
    cuda_dynamics_KS_3d_variable_v_batched as dynamics_variable_torch
)

class TorchPlannerBase(BaseController, ABC):
    """ Provides shared PyTorch functionality for trajectory planning """
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int):
        super().__init__(controller_config, experiment_config, seed)

        # Setup device
        assert torch.cuda.is_available(), "CUDA is not available"
        self.device = torch.device('cuda')

        self.variable_velocity_mode = self.experiment_config.get('variable_velocity_mode', False)

        # Costmap handling: Store the inputs for the current planning cycle.
        self.current_costmap_np = None
        # Tensors derived from the current costmap (cached during the cycle)
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None

        # General planning parameters (from BaseController)
        self.T_horizon = self.num_steps # Use T_horizon for clarity in planning context

        self._init_footprint_kernel()
        self.set_seeds(self.seed)
    
    def _init_footprint_kernel(self):
        """Initializes the kernel for costmap convolution optimization"""
        self.effective_footprint_size = 2 * (self.robot_footprint_size // 2) + 1

        # The kernel is a square of ones matching the effective footprint size.
        # Shape: (out_channels, in_channels, H, W) = (1, 1, size, size)
        self.footprint_kernel = torch.ones(
            (1, 1, self.effective_footprint_size, self.effective_footprint_size),
            dtype=torch.float32,
            device=self.device
        )
        # Padding required for the convolution (S_eff is always odd)
        self.footprint_padding = self.effective_footprint_size // 2

    def set_seeds(self, seed):
        """Set all random seeds for deterministic execution."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic CUDA operations if possible
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def reset(self):
        """Reset shared planner state."""
        self.current_costmap_np = None
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None
        self.set_seeds(self.seed)

    # =================================================================================
    # Input Processing
    # =================================================================================
    def _process_planner_input(self, planner_input: PlannerInput):
        """
        Processes the input provided by the ROS node and prepares internal state for the planning cycle.
        This replaces the simulation's _prepare_perception_inputs.
        """
        self.current_costmap_np = planner_input.inflated_costmap
        # Invalidate tensors derived from the previous costmap
        self.current_costmap_tensor = None
        self.current_convolved_costmap = None

    # =================================================================================
    # Dynamics and Rollouts
    # =================================================================================
    def _rollout_full_controls_torch(self, controls_full, initial_state):
        """
        General rollout using PyTorch dynamics for variable velocity controls.
        controls_full shape: (T, K, 2) [v, w]
        initial_state shape: (3,) [x, y, theta] (must be a Tensor)
        Returns shape: (T+1, K, 3)
        """
        T, K, _ = controls_full.shape
        state_dim = initial_state.shape[0]

        batch_current_states = initial_state.unsqueeze(0).repeat(K, 1)
        trajectory_states = torch.empty((T + 1, K, state_dim), dtype=torch.float32, device=self.device)
        trajectory_states[0] = batch_current_states

        for step in range(T):
            if self.variable_velocity_mode:
                # Variable velocity mode
                current_controls = controls_full[step] # (K, 2)

                batch_current_states = dynamics_variable_torch(
                    batch_current_states, current_controls, self.dt, self.wheelbase
                )
            else:
                # FIXED VELOCITY mode
                # Shape (K,)
                v = controls_full[step, :, 0]
                # Shape (K, 1)
                steering = controls_full[step, :, 1].unsqueeze(1)

                # Optimized path as velocity is constant in fixed mode
                # We assume noise_sigma[0] is 0 or fixed initialization, so v[0] is representative.
                v_scalar = v[0].item()
                batch_current_states = dynamics_scalar_torch(
                    batch_current_states, steering, self.dt, v_scalar, self.wheelbase
                )

            trajectory_states[step + 1] = batch_current_states
        return trajectory_states

    def _rollout_cuniform_controls_torch(self, controls_steer, initial_state, v_const):
        """
        Specialized rollout for C-Uniform initialization (constant velocity).
        controls_steer shape: (T, K) [w]
        initial_state shape: (3,) [x, y, theta] (must be a Tensor)
        Returns shape: (T+1, K, 3)
        """
        T, K = controls_steer.shape
        state_dim = initial_state.shape[0]

        batch_current_states = initial_state.unsqueeze(0).repeat(K, 1)
        trajectory_states = torch.empty((T + 1, K, state_dim), dtype=torch.float32, device=self.device)
        trajectory_states[0] = batch_current_states

        for step in range(T):
            # Dynamics expects (K, 1) shape for steering.
            current_steerings = controls_steer[step].unsqueeze(1)

            batch_current_states = dynamics_scalar_torch(
                batch_current_states, current_steerings, self.dt, v_const, self.wheelbase
            )
            trajectory_states[step + 1] = batch_current_states
        return trajectory_states

    # =================================================================================
    # Cost Evaluation
    # =================================================================================

    def _calculate_trajectory_costs(self, robot_frame_trajectories: torch.Tensor, goal_tensor: torch.Tensor) -> torch.Tensor:
        """
        (Fully Vectorized) Calculates the total cost for a batch of trajectories in the robot frame.
        Implements the logic of early exit (cost accumulation stops after goal reach or collision).
        robot_frame_trajectories shape: (T+1, K, 3)
        goal_tensor shape: (2,)
        Returns shape: (K,)
        """
        # Input shape: (T+1, K, 3)
        positions = robot_frame_trajectories[..., :2] # (T+1, K, 2)

        # --- Profiling Setup ---
        t_start = time.monotonic()
        profile_data = {}

        # 1. Calculate costs at every timestep (Stateless calculations)
        # 1a. Distance costs (T+1, K)
        # Calculate squared distance to goal for all trajectories at all timesteps.
        t0 = time.monotonic()
        dist_to_goal_squared = torch.sum((positions - goal_tensor)**2, dim=2)
        distance_costs = dist_to_goal_squared * self.dist_weight
        profile_data['cost_dist'] = (time.monotonic() - t0) * 1000

        # 1b. Obstacle costs (T+1, K)
        # Use the optimized, vectorized function which utilizes the pre-convolved costmap.
        t0 = time.monotonic()
        obstacle_costs_raw = self._calculate_robot_frame_costmap_cost(positions)
        profile_data['cost_obs_lookup'] = (time.monotonic() - t0) * 1000


        # 2. Determine stateful masks (Collision and Goal Reached)
        # 2a. Collision detection at each timestep (T+1, K)
        t0 = time.monotonic()
        collision_threshold = self.robot_footprint_area * self.collision_occupancy_ratio
        # Boolean mask indicating if a collision occurred exactly at time t.
        collided_at_t = obstacle_costs_raw > collision_threshold

        # 2b. Goal reached detection at each timestep (T+1, K)
        goal_tolerance_squared = self.goal_tolerance ** 2
        # Boolean mask indicating if the goal was reached exactly at time t.
        reached_at_t = dist_to_goal_squared <= goal_tolerance_squared

        # 3. Propagate masks forward in time (Stateful logic vectorization)
        # If an event occurs at time t, the mask must remain true for all t' > t.
        # use cumulative maximum (cummax) along the time dimension (dim=0)
        # This acts as a cumulative OR operation. Using .byte() is efficient.

        # (T+1, K). True if collided at or before t
        # .values extracts the tensor from the (values, indices) tuple returned by cummax
        collision_mask = torch.cummax(collided_at_t.byte(), dim=0).values.bool()
        # (T+1, K). True if reached goal at or before t
        goal_reached_mask = torch.cummax(reached_at_t.byte(), dim=0).values.bool()
        profile_data['cost_mask_prop'] = (time.monotonic() - t0) * 1000

        # 4. Apply masks to costs
        t0 = time.monotonic()
        # Combined termination mask (T+1, K)
        is_terminated = collision_mask | goal_reached_mask

        # 4a. Apply termination mask to distance costs
        # If terminated (collided or reached goal) at or before t, the distance cost at t is zeroed out
        distance_costs_masked = torch.where(
            is_terminated,
            torch.zeros_like(distance_costs),
            distance_costs
        )

        # 4b. Process and apply masks to obstacle costs
        # Calculate the scaled obstacle cost (used when not in a hard collision)
        scaled_obstacle_costs = obstacle_costs_raw * (self.obs_penalty / self.robot_footprint_area)

        # Apply collision mask: If collided at or before t, use full penalty, otherwise use scaled cost
        obstacle_costs_penalized = torch.where(
            collision_mask,
            torch.full_like(scaled_obstacle_costs, self.obs_penalty),
            scaled_obstacle_costs
        )

        # Apply goal reached mask: If goal reached at or before t, zero out obstacle cost (early exit)
        obstacle_costs_masked = torch.where(
            goal_reached_mask,
            torch.zeros_like(obstacle_costs_penalized),
            obstacle_costs_penalized
        )
        profile_data['cost_mask_apply'] = (time.monotonic() - t0) * 1000

        # 5. Calculate Total Cost (Sum over time dimension)
        # (T+1, K) -> (K,)
        t0 = time.monotonic()
        total_costs = torch.sum(distance_costs_masked + obstacle_costs_masked, dim=0)

        # 6. Terminal costs
        # Terminal costs apply only for trajectories that never reached the goal AND were always safe.

        # The final state of the masks (at T) tells us if the event ever happened.
        ever_collided = collision_mask[-1] # (K,)
        ever_reached_goal = goal_reached_mask[-1] # (K,)

        is_safe_and_unsuccessful = ~(ever_reached_goal | ever_collided)

        # Calculate terminal distance based on the very last position (already calculated)
        terminal_distances_squared = dist_to_goal_squared[-1]
        terminal_costs = is_safe_and_unsuccessful.float() * terminal_distances_squared * self.terminal_weight
        total_costs += terminal_costs
        profile_data['cost_sum_terminal'] = (time.monotonic() - t0) * 1000
        profile_data['cost_total'] = (time.monotonic() - t_start) * 1000
        # Attach profile data to the costs tensor itself for access in the calling function
        # This avoids changing the function signature.
        if not hasattr(total_costs, '_profile_data'):
            setattr(total_costs, '_profile_data', {})
        total_costs._profile_data.update(profile_data)
        return total_costs

    def _ensure_convolved_costmap(self):
        """ Ensures the costmap tensor is loaded and pre-convolved with the robot footprint """
        # Check if the raw costmap needs updating or loading
        if self.current_costmap_tensor is None and self.current_costmap_np is not None:
            self.current_costmap_tensor = torch.from_numpy(self.current_costmap_np).float().to(self.device)
            # Perform 2D convolution
            # Input shape: (H, W) -> (1, 1, H, W)
            input_map = self.current_costmap_tensor.unsqueeze(0).unsqueeze(0)

            # --- Boundary Conditions ---
            # used index clamping when the footprint extended beyond the map bounds, equivalent to 'replication' padding
            # manually pad the input map before convolution
            P = self.footprint_padding
            padded_input_map = F.pad(input_map, (P, P, P, P), mode='replicate')

            # Apply the convolution using the pre-initialized kernel with padding=0 ('valid')
            convolved_output = F.conv2d(
                padded_input_map,
                self.footprint_kernel,
                padding=0
            )

            # (1, 1, H, W) -> (H, W)
            self.current_convolved_costmap = convolved_output.squeeze(0).squeeze(0)
        
        # Handle the case where no map is available
        if self.current_costmap_np is None:
            self.current_costmap_tensor = None
            self.current_convolved_costmap = None

    def _calculate_robot_frame_costmap_cost(self, robot_frame_positions: torch.Tensor) -> torch.Tensor:
        """
        (Vectorized) Calculate obstacle costs using the pre-convolved local costmap.
        Input shape: (..., 2) (e.g., (T+1, K, 2) or (K, 2))
        Output shape: (...)
        """
        # Ensure the costmap is loaded and convolved
        self._ensure_convolved_costmap()

        if self.current_convolved_costmap is None:
            # Return zero costs if no costmap loaded
            return torch.zeros(robot_frame_positions.shape[:-1], device=self.device)

        # The convolved costmap holds the total footprint cost at each cell
        # perform a fast lookup using vectorized indexing
        resolution = self.local_costmap_resolution

        # Robot frame: robot at center of the grid, facing +X direction
        costmap_size = self.current_convolved_costmap.shape[0]
        center = costmap_size // 2

        # Convert robot frame coordinates (..., 2) to grid indices (...), use .long() for truncation
        grid_x = (center + robot_frame_positions[..., 0] / resolution).long()
        grid_y = (center + robot_frame_positions[..., 1] / resolution).long()
        # grid_y = (center - robot_frame_positions[..., 1] / resolution).long()  # Y-flip for image coordinates

        # Clamp indices to valid range (Conservative approach)
        grid_x = torch.clamp(grid_x, 0, costmap_size - 1)
        grid_y = torch.clamp(grid_y, 0, costmap_size - 1)
    
        # Index the convolved costmap (Vectorized gathering - single GPU operation)
        costs = self.current_convolved_costmap[grid_y, grid_x]
        return costs