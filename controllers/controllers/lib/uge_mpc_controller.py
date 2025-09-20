# controllers/lib/uge_mpc_controller.py
import torch
import logging
import math
import numpy as np
from typing import Tuple, Dict, Any
import time

# Import necessary components from the framework
from .torch_planner_base import TorchPlannerBase, PlannerInput
from .mppi_pytorch_controller import MPPIPyTorchController

class UGEMPCController(TorchPlannerBase):
    """
    Uncertainty-Guided Exploratory MPC (UGE-MPC) Controller (PyTorch Implementation).
    Implements Algorithm 2 from the paper: UGE-TO initialization followed by MPPI refinement.
    This replicates the exact behavior of the original Numba implementation (uae_method_3d_TO.py).
    """
    
    def __init__(self, controller_config: dict, experiment_config: dict, seed: int = None, mppi_config: dict = None, **kwargs):
        
        if seed is None:
            seed = experiment_config.get('seed', 2025) if experiment_config else 2025

        # Initialize base class (TorchPlannerBase and BaseController)
        super().__init__(controller_config, experiment_config, seed)
        
        self.logger = logging.getLogger(self.__class__.__name__)

        # Standardized dimensions
        self.T = self.T_horizon
        self.nu = 2 # Control dimension [v, delta]
        self.nx = 3 # State dimension [x, y, theta]
        
        # Load UGE-MPC specific parameters and initialize components
        self._load_uge_params()
        self._initialize_components(experiment_config, seed, mppi_config)

        # Initialize the nominal control sequence (maintained across MPC steps)
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        self.U_nominal[:, 0] = float(self.vrange[0])

        self.logger.info(f"UGEMPCController initialized. UGE-TO (N={self.N}, M={self.M}, Iters={self.iters}), MPPI (L={self.L})")

    def _load_uge_params(self):
        """
        Loads parameters from the config, defining UGE-TO, MPPI, and Noise models.
        """
        try:
            # --- UGE-TO Parameters (Algorithm 1) ---
            uge_to_cfg = self.config["uge_to"]
            self.N = uge_to_cfg["num_trajectories"]
            self.M = uge_to_cfg["candidates_per_traj"]
            self.iters = uge_to_cfg["iterations"]
            self.step_interval = uge_to_cfg.get("step_interval", 5)
            self.decay_sharpness = uge_to_cfg.get("decay_sharpness", 2.0)
            
            # Pre-calculate Hellinger indices
            self.hellinger_indices = torch.arange(0, self.T + 1, self.step_interval, dtype=torch.long, device=self.device)
            self.S = len(self.hellinger_indices)

            # Pre-calculate decay coefficients (matches Numba: exp(linspace(log(2.0), log(1.0), iters) ** sharpness))
            if self.iters > 0:
                log_start, log_end = np.log(2.0), np.log(1.0)
                linspace = torch.linspace(log_start, log_end, self.iters, device=self.device, dtype=torch.float32)
                self.decay_coeffs = torch.exp(torch.pow(linspace, self.decay_sharpness))
            else:
                self.decay_coeffs = torch.tensor([], device=self.device)

            # --- Noise Parameters ---
            noise_cfg = self.config["noise"]
            
            # R (Sigma_u): Covariance for sampling perturbations
            R_std = np.array(noise_cfg["R_std"], dtype=np.float32)
            self.R_cov = torch.diag(torch.tensor(R_std**2, device=self.device, dtype=torch.float32))
            
            # Q: Covariance for EKF propagation (Input noise model BQB^T)
            Q_std = np.array(noise_cfg["Q_std"], dtype=np.float32)
            self.Q_cov = torch.diag(torch.tensor(Q_std**2, device=self.device, dtype=torch.float32))

            # Sigma0: Initial state covariance
            Sigma0_std = np.array(noise_cfg["Sigma0_std"], dtype=np.float32)
            self.Sigma0_cov = torch.diag(torch.tensor(Sigma0_std**2, device=self.device, dtype=torch.float32))

            # --- MPPI Parameters (Algorithm 2, Stage 3) ---
            mppi_cfg = self.config["mppi"]
            self.L = mppi_cfg["num_rollouts"]
            # Store the specific configuration intended for the refinement stage
            self.mppi_refinement_config = mppi_cfg.get("refinement_config")

        except (KeyError, ValueError) as e:
            self.logger.error(f"CRITICAL ERROR during parameter loading: {e}")
            raise RuntimeError(f"Configuration invalid or incomplete. Error: {e}.")

    def _initialize_components(self, experiment_config, seed, mppi_config_override):
        """Initialize the MPPI refiner and pre-calculate Cholesky decompositions."""
        
        # Determine the MPPI configuration to use. Prioritize the specific 'refinement_config' block.
        # If 'refinement_config' is missing, use the 'mppi_config' passed from the ROS node factory (which usually points to the general 'mppi_controller' block).
        config_to_use = self.mppi_refinement_config if self.mppi_refinement_config is not None else mppi_config_override
        
        if config_to_use is None:
             raise ValueError("MPPI configuration missing. Ensure 'refinement_config' is in YAML or 'mppi_config' is passed via the ROS node factory.")

        # Ensure the config has the required 'num_rollouts' key for BaseController initialization
        if 'num_rollouts' not in config_to_use:
            config_to_use['num_rollouts'] = self.L

        # Initialize MPPI Refiner
        self.mppi_refiner = MPPIPyTorchController(
            controller_config=config_to_use,
            experiment_config=experiment_config,
            type_override=0, # UGE-MPC uses standard Gaussian MPPI (Type 0)
            seed=seed
        )
        self.mppi_refiner.K = self.L # Ensure budget matches L

        # Pre-calculate Cholesky decompositions for efficient sampling
        try:
            # Use double precision for stability, then cast back to float.
            self.chol_R = torch.linalg.cholesky(self.R_cov.double()).float().to(self.device)
            # Numba implementation uses 3*R specifically for the initial sampling in optimize3D
            self.chol_3R = torch.linalg.cholesky((3 * self.R_cov).double()).float().to(self.device)
        except torch.linalg.LinAlgError as e:
            self.logger.error(f"Cholesky decomposition failed for R_cov. Ensure R_std values create a positive definite matrix. Error: {e}")
            raise

    # Implement the required abstract method
    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action using UGE-MPC (Algorithm 2)."""
        
        # 0. Setup and Warm-start
        self._process_planner_input(planner_input)
        self._shift_nominal_trajectory()
        U_start = self.U_nominal.clone()

        # Robot frame setup
        x0_robot_frame = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        goal_tensor = torch.from_numpy(planner_input.local_goal).float().to(self.device)

        # 1. UGE-TO Initialization (Algorithm 2, Stage 1 & 2)
        # Runs Alg. 1 and selects the best trajectory i*
        U_nominal_ugto, uge_to_trajs_T, best_idx_ugto = self._run_uge_to_initialization(
            U_start, x0_robot_frame, goal_tensor
        )

        # 2. MPPI Refinement (Algorithm 2, Stage 3)
        # Ensure the MPPI refiner uses the same perception data.
        self.mppi_refiner._process_planner_input(planner_input)

        # Run the optimization routine from the MPPI instance, starting from the UGE-TO result.
        # mppi_trajectories_T shape: (T+1, L, nx)
        U_nominal_final, mppi_trajectories_T = self.mppi_refiner._run_mppi_optimization(
            U_nominal_ugto, planner_input.local_goal
        )

        # Update the internal state for the next iteration
        self.U_nominal = U_nominal_final

        # 3. Final Control Selection
        control_action_np = U_nominal_final[0].cpu().numpy()

        # 4. Visualization (Hybrid 4-color)
        if self.viz_config.get('enabled', True) and self.viz_config.get('visualize_trajectories', True):
            info = self._prepare_visualization_info_hybrid(
                uge_to_trajs_T, best_idx_ugto,
                mppi_trajectories_T, U_nominal_final
            )
        else:
            info = {
                'state_rollouts_robot_frame': None,
                # Must be True to indicate the controller type, even if visualization is off
                'is_hybrid': True, 
            }
        
        return control_action_np, info

    def reset(self):
        """Reset controller state."""
        super().reset() 
        self.mppi_refiner.reset()
        self.U_nominal = torch.zeros((self.T, self.nu), dtype=torch.float32, device=self.device)
        self.U_nominal[:, 0] = float(self.vrange[0])

    def _shift_nominal_trajectory(self):
        """Shift the nominal trajectory forward one step (Warm-starting)."""
        self.U_nominal = torch.roll(self.U_nominal, shifts=-1, dims=0)

    # =================================================================================
    # Core UGE-TO Logic (Algorithm 1 Implementation)
    # =================================================================================

    def _run_uge_to_initialization(self, U_nominal_initial, initial_state, goal_tensor):
        """
        Executes UGE-TO (Algorithm 1) and selects the best trajectory (Algorithm 2, Stage 1 & 2).
        Replicates the logic of Numba's optimize3D function.
        """
        
        # 1. Initialization (Numba: optimize3D initialization phase)
        # Initialize N trajectories: 1 base + (N-1) perturbed.
        # Crucial Detail: Numba implementation uses 3*R covariance for this initial sampling.
        action_seqs = torch.empty(self.N, self.T, self.nu, device=self.device, dtype=torch.float32)
        action_seqs[0] = U_nominal_initial
        
        if self.N > 1:
            # Sample noise (N-1, T, nu) using chol_3R
            noise = self._sample_noise_uge_to(self.N - 1, self.T, self.chol_3R)
            action_seqs[1:] = U_nominal_initial.unsqueeze(0) + noise
        
        # Clamp initial actions
        action_seqs = self._clamp_controls(action_seqs)

        # Helper indices for "others" (used in Hellinger calculation)
        # others_idx[i] contains all indices [0..N-1] excluding i.
        others_idx = [torch.tensor([j for j in range(self.N) if j != i], dtype=torch.long, device=self.device)
                      for i in range(self.N)]

        # 2. Iterative Distributional Separation (Numba: optimize3D loop)
        for iter_idx in range(self.iters):
            
            # 2a. Propagate Mean and Covariance
            trajs, covs = self._propagate_mean_and_covariance(initial_state, action_seqs)
            
            # Extract Gaussians at specific intervals (S)
            means3d, covs3d = self._extract_gaussians_at_idx3d(trajs, covs, self.hellinger_indices)

            # 2b. Generate M Candidates per Trajectory (for i=1 to N-1)
            # Crucial Detail: Numba implementation refines only N-1 trajectories, keeping index 0 fixed during iterations.
            N_minus_1 = self.N - 1
            if N_minus_1 == 0: break

            C_total = N_minus_1 * self.M
            
            # Get the decay coefficient for this iteration
            decay_coeff = self.decay_coeffs[iter_idx]
            
            # Sample noise (C_total, T, nu) using chol_R (standard R used here) and apply decay
            noise_candidates = self._sample_noise_uge_to(C_total, self.T, self.chol_R) * decay_coeff
            
            # Repeat base action sequences (i=1 to N-1) M times: (N-1, T, nu) -> (C_total, T, nu)
            action_seqs_repeated = action_seqs[1:].repeat_interleave(self.M, dim=0)
            
            cand_all_flat = action_seqs_repeated + noise_candidates
            cand_all_flat = self._clamp_controls(cand_all_flat)

            # 2c. Propagate Candidates
            trajs_c_all, covs_c_all = self._propagate_mean_and_covariance(initial_state, cand_all_flat)
            mu_c_all, Si_c_all = self._extract_gaussians_at_idx3d(trajs_c_all, covs_c_all, self.hellinger_indices)

            # 2d. Score and Select
            # This step calculates Hellinger distance and selects the best candidate for trajectories i=1 to N-1.
            selected_actions_N_minus_1 = self._score_and_select_pytorch(mu_c_all, Si_c_all, means3d, covs3d,
                                                                        cand_all_flat, others_idx)
            
            # Update the action sequences (keeping index 0 unchanged)
            action_seqs[1:] = selected_actions_N_minus_1

        # 3. Final Evaluation and Selection (Algorithm 2, Stage 2)
        # Re-propagate the final diverse set (ensures consistency for the final evaluation/visualization)
        final_trajs, _ = self._propagate_mean_and_covariance(initial_state, action_seqs)

        # Calculate costs
        # Transpose final_trajs: (N, T+1, nx) -> (T+1, N, nx) for cost calculation
        final_trajs_transposed = final_trajs.permute(1, 0, 2)
        task_costs = self._calculate_trajectory_costs(final_trajs_transposed, goal_tensor)
        
        # Select the best trajectory (i*)
        best_idx = torch.argmin(task_costs).item()
        U_nominal_ugto = action_seqs[best_idx]

        # Return the selected nominal, the set of final trajectories (T+1, N, nx), and the best index
        return U_nominal_ugto, final_trajs_transposed, best_idx

    def _sample_noise_uge_to(self, batch_size, T, chol_R):
        """Samples noise using Cholesky decomposition (Z @ L^T)."""
        Z = torch.randn(batch_size, T, self.nu, device=self.device, dtype=torch.float32)
        # (B, T, k) @ (l, k) -> (B, T, l)
        noise = torch.einsum('btk,lk->btl', Z, chol_R.T)
        return noise

    def _propagate_mean_and_covariance(self, initial_state, action_seqs):
        """
        Propagates the mean state (using TorchPlannerBase dynamics) and the covariance 
        (using PyTorch EKF implementation).
        """
        # 1. Propagate Mean (Standard rollout)
        # Transpose action_seqs: (B, T, nu) -> (T, B, nu)
        action_seqs_transposed = action_seqs.permute(1, 0, 2)
        # (T+1, B, nx)
        trajs_transposed = self._rollout_full_controls_torch(action_seqs_transposed, initial_state)
        # (B, T+1, nx)
        trajs = trajs_transposed.permute(1, 0, 2)

        # 2. Propagate Covariance (EKF style)
        covs = self._propagate_covariance_pytorch(trajs, action_seqs)
        
        return trajs, covs

    def _propagate_covariance_pytorch(self, trajs, actions_batch):
        """
        Efficient, batched PyTorch implementation of the EKF covariance update.
        Matches propagate_uncertainty_batch_numba_fast.
        Update rule: Sigma_{t+1} = A_t Sigma_t A_t^T + B_t Q B_t^T
        """
        B, T, _ = actions_batch.shape
        L = self.wheelbase
        dt = self.dt
        
        # Initialize covariances storage (B, T+1, nx, nx)
        covs = torch.zeros(B, T + 1, self.nx, self.nx, device=self.device, dtype=torch.float32)
        covs[:, 0] = self.Sigma0_cov

        # Extract components for Jacobian calculation (Batched over B and T)
        V = actions_batch[:, :, 0]
        Delta = actions_batch[:, :, 1]
        Theta = trajs[:, :T, 2] # State at time t

        # Pre-calculate trigonometric functions
        SinTh = torch.sin(Theta)
        CosTh = torch.cos(Theta)
        TanDelta = torch.tan(Delta)
        CosDeltaSq = torch.cos(Delta)**2

        # --- Calculate Jacobians A and B (Batched) ---
        
        # Jacobian A (State transition) (B, T, nx, nx)
        # A_t = I + df/dx * dt (Discrete time approximation)
        A = torch.eye(self.nx, device=self.device).view(1, 1, self.nx, self.nx).repeat(B, T, 1, 1)
        A[:, :, 0, 2] = -dt * V * SinTh
        A[:, :, 1, 2] = dt * V * CosTh

        # Jacobian B (Control transition) (B, T, nx, nu)
        # B_t = df/du * dt
        B_mat = torch.zeros(B, T, self.nx, self.nu, device=self.device, dtype=torch.float32)
        B_mat[:, :, 0, 0] = dt * CosTh
        B_mat[:, :, 1, 0] = dt * SinTh
        B_mat[:, :, 2, 0] = dt * TanDelta / L
        # Add clamp for stability if delta is near pi/2
        B_mat[:, :, 2, 1] = dt * V / (L * torch.clamp(CosDeltaSq, min=1e-6))

        # Pre-calculate Noise term B @ Q @ B^T (B, T, nx, nx)
        # (B, T, nx, nu) @ (nu, nu) -> (B, T, nx, nu)
        BQ = torch.matmul(B_mat, self.Q_cov)
        # (B, T, nx, nu) @ (B, T, nu, nx) -> (B, T, nx, nx)
        Noise_term_all = torch.matmul(BQ, B_mat.transpose(-1, -2))

        # EKF Update Loop (Sequential dependency on time)
        current_Sigma = covs[:, 0].clone() # (B, nx, nx)

        for t in range(T):
            At = A[:, t] # (B, nx, nx)
            Noise_term = Noise_term_all[:, t] # (B, nx, nx)
            
            # Sigma = A @ Sigma @ A^T + Noise
            # Use torch.bmm for efficient batched matrix multiplication
            Sigma_prop = torch.bmm(At, torch.bmm(current_Sigma, At.transpose(1, 2)))
            
            current_Sigma = Sigma_prop + Noise_term
            covs[:, t+1] = current_Sigma

        return covs

    def _extract_gaussians_at_idx3d(self, trajs, covs, idx):
        """Extracts means and covariances at specific time indices."""
        # Use torch.index_select along the time dimension (dim=1)
        means3d = torch.index_select(trajs, 1, idx) # (B, S, nx)
        covs3d = torch.index_select(covs, 1, idx)   # (B, S, nx, nx)
        return means3d, covs3d

    def _score_and_select_pytorch(self, mu_c_all, Si_c_all, means3d, covs3d, cand_all_flat, others_idx):
        """
        Calculates Hellinger distances and selects the most diverse candidates for i=1 to N-1.
        Replicates the logic in Numba's score_and_select_fast_3d adapted for the N-1 structure.
        """
        N = self.N
        M = self.M
        N_minus_1 = N - 1
        
        # Initialize tensor to store the selected actions for the N-1 trajectories
        selected_actions = torch.zeros(N_minus_1, self.T, self.nu, device=self.device, dtype=torch.float32)

        # Loop over the N-1 trajectories to update (i=1 to N-1)
        for i in range(1, N):
            # Determine the slice of candidates corresponding to trajectory i
            start_idx = (i-1) * M
            end_idx = start_idx + M

            # Candidates for trajectory i: (M, S, ...)
            mu_c = mu_c_all[start_idx:end_idx]
            Si_c = Si_c_all[start_idx:end_idx]
            
            # The set of other trajectories (N-1, S, ...)
            other_indices = others_idx[i]
            mu_o = means3d[other_indices]
            Si_o = covs3d[other_indices]

            # Calculate Hellinger scores (M,)
            # Score[m] = Sum_{o, s} H^2(Candidate_m, Other_o, Step_s)
            hellinger_scores = self._hellinger_3d_batch_pytorch(mu_c, Si_c, mu_o, Si_o)

            # Select the candidate that maximizes the score (most diverse)
            best_m_idx_local = torch.argmax(hellinger_scores)
            
            # Get the global index in the flattened candidate list
            best_m_idx_global = start_idx + best_m_idx_local
            
            # Store the selected action sequence (index i-1 because we are storing N-1 actions)
            selected_actions[i-1] = cand_all_flat[best_m_idx_global]

        return selected_actions

    def _hellinger_3d_batch_pytorch(self, mu_c, Sig_c, mu_o, Sig_o):
        """
        Robust PyTorch implementation of Hellinger distance calculation.
        Uses log-space calculations (slogdet) and torch.linalg.solve for stability.
        
        Shapes:
          mu_c: (C, S, nx),  Sig_c: (C, S, nx, nx)
          mu_o: (K, S, nx),  Sig_o: (K, S, nx, nx)
        Returns:
          scores: (C,) where score[c] = Sum_{k, s} H^2(c, k, s)
        """
        C, S, nx = mu_c.shape
        K = mu_o.shape[0]

        # Expand dimensions for broadcasting (C, 1, S, ...) vs (1, K, S, ...)
        mu_c_exp = mu_c.unsqueeze(1)
        mu_o_exp = mu_o.unsqueeze(0)
        Sig_c_exp = Sig_c.unsqueeze(1)
        Sig_o_exp = Sig_o.unsqueeze(0)
        
        # 1. Mean differences (Delta_mu) (C, K, S, nx)
        delta_mu = mu_c_exp - mu_o_exp
        
        # Handle angle wrapping for theta (nx=2). Normalize to [-pi, pi].
        # The Numba implementation (hellinger_3d_batch_numba_fast) does not explicitly wrap angles here, 
        # but robust Hellinger calculation requires correct angle differences (e.g., diff between 3.14 and -3.14 is near 0).
        dth = delta_mu[..., 2]
        # Use atan2 for robust angle wrapping
        dth = torch.atan2(torch.sin(dth), torch.cos(dth))
        delta_mu[..., 2] = dth

        # 2. Average Covariance (A = (C+O)/2) (C, K, S, nx, nx)
        Avg_Sig = 0.5 * (Sig_c_exp + Sig_o_exp)

        # 3. Determinants (using slogdet for numerical stability)
        # slogdet returns (sign, log_abs_det)
        _, log_det_c = torch.linalg.slogdet(Sig_c_exp)
        _, log_det_o = torch.linalg.slogdet(Sig_o_exp)
        sign_avg, log_det_avg = torch.linalg.slogdet(Avg_Sig)

        # Check for non-positive definite matrices (det <= 0). 
        # If det_avg is not positive, the distance is maximal (1.0)
        invalid_mask = (sign_avg <= 0) | (log_det_avg.isnan())

        # 4. Pre-factor calculation (in log space)
        # log_pref = 0.25 * (log(|C|) + log(|O|)) - 0.5 * log(|A|)
        log_pref = 0.25 * (log_det_c + log_det_o) - 0.5 * log_det_avg
        
        # 5. Quadratic form (Mahalanobis distance): delta_mu^T @ A^-1 @ delta_mu
        
        # Use torch.linalg.solve for A^-1 @ delta_mu.
        # We must handle invalid matrices before solve to prevent errors/NaNs.
        # Replace invalid Avg_Sig with identity; they will be masked out later.
        Avg_Sig_stable = Avg_Sig.clone()
        # Identify invalid locations (C, K, S) and set corresponding (nx, nx) block to Identity
        Avg_Sig_stable[invalid_mask] = torch.eye(nx, device=self.device)

        try:
            # Solve Ax=b -> x = A^-1 b
            # We use unsqueeze(-1) to treat delta_mu as column vectors for the solve operation
            solved = torch.linalg.solve(Avg_Sig_stable, delta_mu.unsqueeze(-1)).squeeze(-1)
        except torch.linalg.LinAlgError:
            # Fallback if solve fails (e.g., near-singular matrices despite replacement)
            self.logger.warning("torch.linalg.solve failed during Hellinger calculation. Attempting pseudo-inverse.")
            try:
                # Attempt pseudo-inverse as a last resort
                pinv_Avg_Sig = torch.linalg.pinv(Avg_Sig_stable)
                # Use matmul for the pseudo-inverse application
                solved = torch.matmul(pinv_Avg_Sig, delta_mu.unsqueeze(-1)).squeeze(-1)
            except Exception as e:
                 self.logger.error(f"Fallback pseudo-inverse also failed: {e}. Setting distances to max.")
                 # If all fails, return the maximum possible score (K*S)
                 return torch.full((C,), float(K * S), device=self.device)

        # Calculate quadratic form: delta_mu^T @ solved
        # Use einsum for efficient batched dot product: 'cksi,cksi->cks'
        quad = torch.einsum('cksi,cksi->cks', delta_mu, solved)

        # 6. Exponent
        exponent = -0.125 * quad

        # 7. Calculate Bhattacharyya Coefficient (BC) = exp(log_pref + exponent)
        # Clamp exponent as in Numba code for stability (avoid exp(-large_number) becoming zero unnecessarily)
        exponent_clamped = torch.clamp(exponent, min=-60.0)
        log_BC = log_pref + exponent_clamped
        BC = torch.exp(log_BC)

        # 8. Hellinger Distance Squared (H^2 = 1 - BC)
        H_sq = 1.0 - BC

        # 9. Apply masks and clamp
        # If det_avg was invalid (mask=True), H^2 must be 1.0
        H_sq[invalid_mask] = 1.0
        
        # Clamp for numerical stability (H^2 must be in [0, 1])
        H_sq = torch.clamp(H_sq, min=0.0, max=1.0)

        # 10. Sum over K (others) and S (steps) to get the total score for each candidate C
        scores = torch.sum(H_sq, dim=(1, 2)) # (C,)

        return scores

    def _clamp_controls(self, controls):
        """Clamps controls (B, T, nu) to the vehicle limits."""
        if not hasattr(self, 'min_ctrl_tensor'):
            self.min_ctrl_tensor = torch.tensor([float(self.vrange[0]), float(self.wrange[0])], device=self.device).view(1, 1, 2)
            self.max_ctrl_tensor = torch.tensor([float(self.vrange[1]), float(self.wrange[1])], device=self.device).view(1, 1, 2)
        
        return torch.max(torch.min(controls, self.max_ctrl_tensor), self.min_ctrl_tensor)

# =================================================================================
    # Visualization
    # =================================================================================
    
    def _prepare_visualization_info_hybrid(self, uge_to_trajectories_T, best_idx_ugto, 
                                           mppi_trajectories_T, u_nominal_final):
        """
        Prepare visualization data for the hybrid approach (4-color visualization).
        Includes padding and separation of best/samples for visualization stability and clarity.
        Inputs are expected in the ROBOT FRAME.
        
        uge_to_trajectories_T: (T+1, N, nx)
        mppi_trajectories_T: (T+1, L, nx)
        """
        
        # Define the target visualization size based on the configuration
        TARGET_VIS_SIZE = self.num_vis_rollouts # e.g., 1000
        
        # Transform inputs to (Batch, T+1, nx) format and ensure contiguous memory layout
        uge_to_trajectories = uge_to_trajectories_T.permute(1, 0, 2).contiguous()
        mppi_trajectories = mppi_trajectories_T.permute(1, 0, 2).contiguous()
        N_total = uge_to_trajectories.shape[0]

        # --- 1. UGE-TO Best Sample - (Corresponds to 'cu_best' - Blue) ---
        # Shape: (T+1, nx)
        if N_total > 0:
            ugto_best_np = uge_to_trajectories[best_idx_ugto].cpu().numpy()
        else:
            # Handle edge case N=0 (should ideally not happen based on config checks)
            ugto_best_np = np.zeros((self.T + 1, self.nx), dtype=np.float32)

        # --- 2. UGE-TO Samples (Excluding Best) - (Corresponds to 'cu_samples' - Grey) ---
        
        # Initialize a full-sized array, pre-filled with the best trajectory (padding)
        ugto_samples_np = np.tile(ugto_best_np, (TARGET_VIS_SIZE, 1, 1))

        # Create a mask to exclude the best index
        mask = torch.ones(N_total, dtype=torch.bool, device=self.device)
        if N_total > 0:
            mask[best_idx_ugto] = False
        
        # Select the remaining (non-best) trajectories
        uge_to_non_best = uge_to_trajectories[mask]

        # Copy the actual non-best samples into the beginning of the padded array
        num_actual_non_best = uge_to_non_best.shape[0]
        num_to_copy = min(TARGET_VIS_SIZE, num_actual_non_best)
        
        if num_to_copy > 0:
            ugto_samples_np[:num_to_copy] = uge_to_non_best[:num_to_copy].cpu().numpy()

        # --- 3. MPPI Final (Weighted Average) - (Corresponds to 'mppi_nominal' - Green) ---
        # Rollout the final nominal control sequence
        robot_frame_initial_state = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        U_nominal_reshaped = u_nominal_final.unsqueeze(1) # (T, 1, nu)
        
        # Use the refiner's rollout function
        nominal_traj_robot = self.mppi_refiner._rollout_full_controls_torch(U_nominal_reshaped, robot_frame_initial_state)
        
        # (T+1, 1, nx) -> (T+1, nx)
        mppi_nominal_np = nominal_traj_robot.squeeze(1).cpu().numpy()

        # --- 4. MPPI Samples - (Corresponds to 'mppi_samples' - Orange) ---
        # Initialize a full-sized array, pre-filled with the nominal trajectory (padding)
        mppi_samples_np = np.tile(mppi_nominal_np, (TARGET_VIS_SIZE, 1, 1))

        # Copy the actual MPPI samples
        num_actual_mppi_samples = mppi_trajectories.shape[0]
        num_to_copy_mppi = min(TARGET_VIS_SIZE, num_actual_mppi_samples)
        if num_to_copy_mppi > 0:
             # MPPI samples already exclude the nominal, so we copy them directly
             mppi_samples_np[:num_to_copy_mppi] = mppi_trajectories[:num_to_copy_mppi].cpu().numpy()

        # --- 5. Package Data ---
        # We reuse the visualization keys from CU-MPPI for consistency in the ROS node.
        trajectory_data = {
            'cu_samples': ugto_samples_np,
            'cu_best': ugto_best_np,
            'mppi_samples': mppi_samples_np,
            'mppi_nominal': mppi_nominal_np,
        }
        
        # Apply angle wrapping and ensure contiguity (Robustness for ROS serialization)
        for key in trajectory_data:
            traj_array = trajectory_data[key]
            if traj_array is not None:
                # Ensure the array is contiguous (np.tile ensures this, but we double check)
                if not traj_array.flags['C_CONTIGUOUS']:
                    traj_array = np.ascontiguousarray(traj_array)
                
                # Apply wrapping to the theta dimension (index 2).
                # Handles both (N, T, 3) and (T, 3) shapes using ellipsis (...).
                traj_array[..., 2] = np.arctan2(np.sin(traj_array[..., 2]), np.cos(traj_array[..., 2]))
                
                trajectory_data[key] = traj_array

        vis_data = {
            # The ROS node expects this specific dictionary structure when is_hybrid=True
            'state_rollouts_robot_frame': trajectory_data,
            'is_hybrid': True,
        }
        return vis_data