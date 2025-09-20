# controllers/lib/uge_mpc_controller.py
import torch
import logging
import math
import numpy as np
from typing import Tuple, Dict, Any
import time

# Import necessary components from the framework
# Note: Assuming these relative imports work in the user's environment
from .torch_planner_base import TorchPlannerBase, PlannerInput
from .mppi_pytorch_controller import MPPIPyTorchController

# =================================================================================
# JIT Compiled Optimization Kernels (Replicating Numba optimizations)
# =================================================================================

# OPTIMIZATION: Renamed from _hellinger_3d_batch_jit_unrolled.
# Modified to return the full H_sq tensor (C, K, S) for use in vectorized scoring.
@torch.jit.script
def _hellinger_3d_kernel_internal_jit(mu_c: torch.Tensor, Sig_c: torch.Tensor, mu_o: torch.Tensor, Sig_o: torch.Tensor) -> torch.Tensor:
    """
    Fast, JIT-compiled 3D Hellinger^2 calculation using manual 3x3 determinant/inverse.
    This mirrors the optimized Numba implementation (hellinger_3d_batch_numba_fast).
    
    Inputs:
      mu_c: (C, S, nx), Sig_c: (C, S, nx, nx)
      mu_o: (K, S, nx), Sig_o: (K, S, nx, nx)
    Returns:
      H_sq: (C, K, S) - The squared Hellinger distance for every combination.
    """
    # Epsilon matching Numba implementation's float32 precision considerations
    eps = torch.tensor(1e-9, dtype=torch.float32, device=mu_c.device) 

    # Expand dimensions for broadcasting (C, 1, S, ...) vs (1, K, S, ...)
    mu_c_exp = mu_c.unsqueeze(1)
    mu_o_exp = mu_o.unsqueeze(0)
    Sig_c_exp = Sig_c.unsqueeze(1)
    Sig_o_exp = Sig_o.unsqueeze(0)

    # 1. Mean differences (Delta_mu) (C, K, S, nx)
    delta_mu = mu_c_exp - mu_o_exp
    
    # NOTE: We intentionally skip explicit angle wrapping (atan2) here to exactly match 
    # the behavior of the Numba ground truth (uae_method_3d_TO.py), which also omits it.

    # 2. Average Covariance (A = (C+O)/2) (C, K, S, nx, nx)
    Avg_Sig = 0.5 * (Sig_c_exp + Sig_o_exp)

    # 3. Determinant and Inverse of A (Manually unrolled 3x3)
    # Extract elements of A
    a00 = Avg_Sig[..., 0, 0]; a01 = Avg_Sig[..., 0, 1]; a02 = Avg_Sig[..., 0, 2]
    a10 = Avg_Sig[..., 1, 0]; a11 = Avg_Sig[..., 1, 1]; a12 = Avg_Sig[..., 1, 2]
    a20 = Avg_Sig[..., 2, 0]; a21 = Avg_Sig[..., 2, 1]; a22 = Avg_Sig[..., 2, 2]

    # det(A)
    det_avg = (a00*(a11*a22 - a12*a21)
              -a01*(a10*a22 - a12*a20)
              +a02*(a10*a21 - a11*a20))

    # Mask for invalid determinants (det <= eps)
    invalid_mask = det_avg <= eps

    # Cofactors for Inverse(A) = Adjugate(A)^T / det(A)
    cof00 =  (a11*a22 - a12*a21)
    cof01 = -(a10*a22 - a12*a20)
    cof02 =  (a10*a21 - a11*a20)

    cof10 = -(a01*a22 - a02*a21)
    cof11 =  (a00*a22 - a02*a20)
    cof12 = -(a00*a21 - a01*a20)

    cof20 =  (a01*a12 - a02*a11)
    cof21 = -(a00*a12 - a02*a10)
    cof22 =  (a00*a11 - a01*a10)

    # Handle division by zero: replace det_avg with 1.0 where invalid (masked later)
    det_avg_safe = torch.where(invalid_mask, torch.ones_like(det_avg), det_avg)

    # Inverse components (Transposed Adjugate / det)
    inv00 = cof00 / det_avg_safe; inv01 = cof10 / det_avg_safe; inv02 = cof20 / det_avg_safe
    inv10 = cof01 / det_avg_safe; inv11 = cof11 / det_avg_safe; inv12 = cof21 / det_avg_safe
    inv20 = cof02 / det_avg_safe; inv21 = cof12 / det_avg_safe; inv22 = cof22 / det_avg_safe

    # 4. Quadratic form: delta_mu^T @ A^-1 @ delta_mu
    dx = delta_mu[..., 0]; dy = delta_mu[..., 1]; dth = delta_mu[..., 2]

    # A^-1 @ delta_mu
    solx = inv00*dx + inv01*dy + inv02*dth
    soly = inv10*dx + inv11*dy + inv12*dth
    solz = inv20*dx + inv21*dy + inv22*dth
    
    # delta_mu^T @ (A^-1 @ delta_mu)
    quad = dx*solx + dy*soly + dth*solz

    # 5. Exponent
    exponent = -0.125 * quad

    # 6. Determinants of C and O (Manually unrolled 3x3)
    # Extract elements of C
    c00 = Sig_c_exp[..., 0, 0]; c01 = Sig_c_exp[..., 0, 1]; c02 = Sig_c_exp[..., 0, 2]
    c10 = Sig_c_exp[..., 1, 0]; c11 = Sig_c_exp[..., 1, 1]; c12 = Sig_c_exp[..., 1, 2]
    c20 = Sig_c_exp[..., 2, 0]; c21 = Sig_c_exp[..., 2, 1]; c22 = Sig_c_exp[..., 2, 2]
    det_c = (c00*(c11*c22 - c12*c21)
            -c01*(c10*c22 - c12*c20)
            +c02*(c10*c21 - c11*c20))

    # Extract elements of O
    o00 = Sig_o_exp[..., 0, 0]; o01 = Sig_o_exp[..., 0, 1]; o02 = Sig_o_exp[..., 0, 2]
    o10 = Sig_o_exp[..., 1, 0]; o11 = Sig_o_exp[..., 1, 1]; o12 = Sig_o_exp[..., 1, 2]
    o20 = Sig_o_exp[..., 2, 0]; o21 = Sig_o_exp[..., 2, 1]; o22 = Sig_o_exp[..., 2, 2]
    det_o = (o00*(o11*o22 - o12*o21)
            -o01*(o10*o22 - o12*o20)
            +o02*(o10*o21 - o11*o20))

    # 7. Calculate Bhattacharyya Coefficient (BC)
    # Clamp exponent for stability (matching Numba behavior)
    exponent_clamped = torch.clamp(exponent, min=-60.0)
    
    # Pre-factor: ((|C|*|O|)^0.25) / sqrt(|A|)
    # Ensure determinants are non-negative before root operations
    det_c_safe = torch.clamp(det_c, min=0.0)
    det_o_safe = torch.clamp(det_o, min=0.0)
    
    # Use det_avg_safe (which is 1.0 where invalid) for the denominator sqrt
    pref = torch.pow(det_c_safe * det_o_safe, 0.25) / torch.sqrt(det_avg_safe)
    
    BC = pref * torch.exp(exponent_clamped)

    # 8. Hellinger Distance Squared (H^2 = 1 - BC)
    H_sq = 1.0 - BC

    # 9. Apply masks and clamp
    # If det_avg was invalid (mask=True), H^2 must be 1.0
    # We use torch.where for JIT compatibility
    H_sq = torch.where(invalid_mask, torch.ones_like(H_sq), H_sq)
    
    # Clamp for numerical stability (H^2 must be in [0, 1])
    H_sq = torch.clamp(H_sq, min=0.0, max=1.0)

    # 10. Return H_sq (C, K, S)
    return H_sq

# OPTIMIZATION: New JIT kernel for vectorized scoring and selection
@torch.jit.script
def _score_and_select_vectorized_jit(
    mu_c_all: torch.Tensor, Si_c_all: torch.Tensor,
    means3d: torch.Tensor, covs3d: torch.Tensor,
    cand_all_flat: torch.Tensor,
    N: int, M: int,
    mask: torch.Tensor
) -> torch.Tensor:
    """
    Fully vectorized and JIT-compiled scoring and selection process.
    Eliminates the Python loop over N-1 trajectories by batching the Hellinger calculation
    and using a mask to exclude self-comparisons. This significantly improves GPU utilization.
    """
    # N: Total number of base trajectories
    # M: Candidates per trajectory

    # 1. Calculate Hellinger distance between all candidates (C_total) and all base trajectories (N).
    # Input mu_c: (C_total, S, 3), Input mu_o: (N, S, 3)
    # Output H_sq shape: (C_total, N, S)
    H_sq = _hellinger_3d_kernel_internal_jit(mu_c_all, Si_c_all, means3d, covs3d)

    # 2. Reshape H_sq to group by the originating trajectory.
    # ( (N-1)*M, N, S ) -> (N-1, M, N, S)
    N_minus_1 = N - 1
    S = H_sq.shape[2]
    H_sq_reshaped = H_sq.view(N_minus_1, M, N, S)

    # 3. Apply the mask to exclude self-comparisons.
    # Mask shape: (N-1, N). Broadcast to (N-1, 1, N, 1).
    # Multiplying zeroes out the H^2 values that should be excluded from the sum.
    H_sq_masked = H_sq_reshaped * mask.unsqueeze(1).unsqueeze(3)

    # 4. Sum over N (others) and S (steps) to get the final scores.
    # Shape: (N-1, M)
    scores = torch.sum(H_sq_masked, dim=(2, 3))

    # 5. Select the candidate that maximizes the score (most diverse) within each group (N-1).
    # Shape: (N-1,)
    best_m_indices_local = torch.argmax(scores, dim=1)

    # 6. Gather the selected actions.
    # Calculate the global index in cand_all_flat: Global index = group_index * M + local_index
    group_indices = torch.arange(N_minus_1, device=mu_c_all.device)
    best_m_indices_global = group_indices * M + best_m_indices_local

    # Gather the corresponding action sequences.
    # Shape: (N-1, T, nu)
    selected_actions = cand_all_flat[best_m_indices_global]
    return selected_actions

@torch.jit.script
def _propagate_covariance_jit_optimized(
    T: int, B: int, nx: int,
    a02_T: torch.Tensor, # (T, B)
    a12_T: torch.Tensor, # (T, B)
    BQB_T: torch.Tensor, # (T, B, nx, nx)
    Sigma0_batch: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    JIT-compiled EKF propagation loop using explicit element-wise expansion (A=I+E).
    Optimized memory layout (T, B, ...) and removal of clones inside the loop.
    """
    # Initialize covariances storage (T+1, B, nx, nx) for efficient access
    covs = torch.zeros(T + 1, B, nx, nx, device=device, dtype=torch.float32)
    covs[0] = Sigma0_batch

    # current_Sigma (S) initialization (B, nx, nx). Clone is crucial.
    S = Sigma0_batch.clone()

    # Optimized Update Loop (JIT Compiled) - Optimized memory layout and no clones inside loop
    for t in range(T):
        # Extract Jacobians and Noise for time t (B,). Contiguous access.
        A02_t = a02_T[t]
        A12_t = a12_T[t]
        BQB_t = BQB_T[t]

        # Load S components (Views). We assume symmetry S01=S10, S02=S20, S12=S21.
        S00 = S[:, 0, 0]; S01 = S[:, 0, 1]; S02 = S[:, 0, 2]
        S11 = S[:, 1, 1]; S12 = S[:, 1, 2]
        S22 = S[:, 2, 2]

        # Calculate N components (New Tensors, NO CLONE, NO IN-PLACE modification of S yet)
        
        # --- ES terms (using symmetry S20=S02, S21=S12) ---
        # E = [[0, 0, a02], [0, 0, a12], [0, 0, 0]]
        ES00 = A02_t * S02; ES01 = A02_t * S12; ES02 = A02_t * S22
        # ES10 = A12_t * S02; # Symmetric to SET10
        ES11 = A12_t * S12; ES12 = A12_t * S22
        # ES2x = 0

        # --- SE^T terms (using symmetry) ---
        SET00 = S02 * A02_t; SET01 = S02 * A12_t; # SET02 = 0
        # SET10 = S12 * A02_t; # Symmetric to ES01
        SET11 = S12 * A12_t; # SET12 = 0
        # SET2x = ... (Calculated below during combination)

        # --- ESE^T terms (only top-left 2x2) ---
        # Optimization: Pre-calculate intermediate products
        A02_S22 = A02_t * S22
        A12_S22 = A12_t * S22

        ESET00 = A02_t * A02_S22
        ESET01 = A12_t * A02_S22
        # ESET10 = Symmetric
        ESET11 = A12_t * A12_S22

        # --- Combine: N = S + ES + SE^T + ESE^T ---
        N00 = S00 + ES00 + SET00 + ESET00
        N01 = S01 + ES01 + SET01 + ESET01
        N02 = S02 + ES02 # + SET02 (is 0)

        N11 = S11 + ES11 + SET11 + ESET11
        N12 = S12 + ES12 # + SET12 (is 0)

        # Combine results back into the matrix S (B, nx, nx)
        S[:, 0, 0] = N00; S[:, 0, 1] = N01; S[:, 0, 2] = N02
        S[:, 1, 0] = N01; S[:, 1, 1] = N11; S[:, 1, 2] = N12 # Enforce symmetry
        # Update the third row (SET2x terms). N20 = S20 + SET20. SET20 = S22*a02.
        S[:, 2, 0] = S02 + A02_S22 # S02 (from S, symmetric S20) + S22*a02 (SET20)
        S[:, 2, 1] = S12 + A12_S22 # S12 (from S, symmetric S21) + S22*a12 (SET21)
        # S[:, 2, 2] remains S22

        # Add Noise term BQB^T
        S += BQB_t

        # Store result for time t+1
        covs[t+1] = S
        
    # Transpose back to (B, T+1, nx, nx)
    return covs.permute(1, 0, 2, 3)

# =================================================================================
# UGEMPCController Class Definition
# =================================================================================
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

        # Profiling configuration - Read from config, default to True if not set (matching original behavior for analysis)
        self.ENABLE_PROFILING = True
        self.profiling_data = {}
        self.profiling_iteration = 0

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

    def _profile_start(self, section_name: str):
        """Start timing a section if profiling is enabled."""
        if self.ENABLE_PROFILING:
            torch.cuda.synchronize()
            return time.perf_counter()
        return None

    def _profile_end(self, section_name: str, start_time):
        """End timing a section and log the result if profiling is enabled."""
        if self.ENABLE_PROFILING and start_time is not None:
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if section_name not in self.profiling_data:
                self.profiling_data[section_name] = []
            self.profiling_data[section_name].append(elapsed_ms)
            
            # Log detailed timing every iteration
            self.logger.info(f"[PROFILE] {section_name}: {elapsed_ms:.2f}ms")
            return elapsed_ms
        return None

    def _profile_log_summary(self):
        """Log a summary of all profiling data for the current iteration."""
        if self.ENABLE_PROFILING and self.profiling_data:
            summary_parts = []
            overall_time = 0.0
            if "Overall" in self.profiling_data and self.profiling_data["Overall"]:
                overall_time = self.profiling_data["Overall"][-1]

            # Updated sections to reflect the optimized structure
            # We show the total time spent in key areas across all iterations.
            sections_to_show = ["UGE-TO_Total", "CombinedPropagation", "HellingerScoring", "MPPI_Total", "Visualization"]

            for section in sections_to_show:
                 if section in self.profiling_data and self.profiling_data[section]:
                    # Calculate the total time spent in this section across all iterations
                    total_section_time = sum(self.profiling_data[section])
                    summary_parts.append(f"{section}: {total_section_time:.2f}ms")
            
            summary_str = " | ".join(summary_parts)
            self.logger.info(f"[PROFILE SUMMARY] Iteration {self.profiling_iteration} - Total: {overall_time:.2f}ms | {summary_str}")
            
            # Clear data for next iteration to prevent memory buildup
            self.profiling_data = {}

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

            # Pre-calculate decay coefficients
            if self.iters > 0:
                log_start, log_end = np.log(2.0), np.log(1.0)
                linspace = torch.linspace(log_start, log_end, self.iters, device=self.device, dtype=torch.float32)
                self.decay_coeffs = torch.exp(torch.pow(linspace, self.decay_sharpness))
            else:
                self.decay_coeffs = torch.tensor([], device=self.device)

            # OPTIMIZATION: Pre-calculate the selection mask for vectorized scoring
            self._initialize_selection_mask()

            noise_cfg = self.config["noise"]
            R_std = np.array(noise_cfg["R_std"], dtype=np.float32)
            self.R_cov = torch.diag(torch.tensor(R_std**2, device=self.device, dtype=torch.float32))
            Q_std = np.array(noise_cfg["Q_std"], dtype=np.float32)
            self.Q_cov = torch.diag(torch.tensor(Q_std**2, device=self.device, dtype=torch.float32))
            Sigma0_std = np.array(noise_cfg["Sigma0_std"], dtype=np.float32)
            self.Sigma0_cov = torch.diag(torch.tensor(Sigma0_std**2, device=self.device, dtype=torch.float32))

            mppi_cfg = self.config["mppi"]
            self.L = mppi_cfg["num_rollouts"]
            self.mppi_refinement_config = mppi_cfg.get("refinement_config")

        except (KeyError, ValueError) as e:
            self.logger.error(f"CRITICAL ERROR during parameter loading: {e}")
            raise RuntimeError(f"Configuration invalid or incomplete. Error: {e}.")

    # OPTIMIZATION: initialization helper
    def _initialize_selection_mask(self):
        """
        Pre-calculates the mask used in vectorized scoring to exclude self-comparisons.
        The mask ensures that candidates generated from trajectory i (i>0) are not compared
        against the base trajectory i itself when calculating the diversity score.
        Mask shape: (N-1, N)
        """
        if self.N <= 1:
            self.selection_mask = torch.empty(0, self.N, device=self.device, dtype=torch.float32)
            return

        N_minus_1 = self.N - 1
        # Initialize mask to all True (1.0)
        mask = torch.ones(N_minus_1, self.N, device=self.device, dtype=torch.float32)

        # The rows correspond to groups g=0..N-2 (trajectories i=1..N-1).
        # The columns correspond to the base trajectories j=0..N-1.
        # We want to set mask[g, j] = 0 where j = g + 1 (the self-comparison).

        # Create indices for the diagonal starting from column 1
        row_indices = torch.arange(N_minus_1, device=self.device)
        col_indices = row_indices + 1

        # Use index_put_ to efficiently set the specific elements to 0.0
        mask.index_put_((row_indices, col_indices), torch.tensor(0.0, device=self.device, dtype=torch.float32))

        self.selection_mask = mask

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
    
    def get_control_action(self, planner_input: PlannerInput) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute optimal control action using UGE-MPC (Algorithm 2)."""
        # 
        # Start overall timing
        overall_start = self._profile_start("Overall")
        # 
        # 0. Setup and Warm-start (~0.9ms - lightweight)
        self._process_planner_input(planner_input)
        self._shift_nominal_trajectory()
        U_start = self.U_nominal.clone()

        # Robot frame setup
        x0_robot_frame = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        goal_tensor = torch.from_numpy(planner_input.local_goal).float().to(self.device)

        # 1. UGE-TO Initialization - MAIN BOTTLENECK (~1044ms)
        ugeto_start = self._profile_start("UGE-TO_Total")
        # Runs Alg. 1 and selects the best trajectory i*
        U_nominal_ugto, uge_to_trajs_T, best_idx_ugto = self._run_uge_to_initialization(
            U_start, x0_robot_frame, goal_tensor
        )
        self._profile_end("UGE-TO_Total", ugeto_start)

        # 2. MPPI Refinement (~14ms - lightweight compared to UGE-TO)
        mppi_start = self._profile_start("MPPI_Total")
        # Ensure the MPPI refiner uses the same perception data.
        self.mppi_refiner._process_planner_input(planner_input)

        # Run the optimization routine from the MPPI instance, starting from the UGE-TO result.
        # mppi_trajectories_T shape: (T+1, L, nx)
        U_nominal_final, mppi_trajectories_T = self.mppi_refiner._run_mppi_optimization(
            U_nominal_ugto, planner_input.local_goal
        )
        self._profile_end("MPPI_Total", mppi_start)

        # Update the internal state for the next iteration
        self.U_nominal = U_nominal_final

        # 3. Final Control Selection (~0.1ms - lightweight)
        control_action_np = U_nominal_final[0].cpu().numpy()

        # 4. Visualization (~15ms total)
        viz_start = self._profile_start("Visualization")
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
        self._profile_end("Visualization", viz_start)
        # End overall timing and log summary
        self._profile_end("Overall", overall_start)
        self.profiling_iteration += 1
        self._profile_log_summary()
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
        (OPTIMIZED) Uses combined batch propagation and vectorized scoring.
        """
        
        # 1. Initialization (Numba: optimize3D initialization phase)
        init_start = self._profile_start("UGE-TO_Init")
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

        self._profile_end("UGE-TO_Init", init_start)

        # 2. Iterative Distributional Separation (Optimized Loop)
        iterations_start = self._profile_start("UGE-TO_Iterations")
        
        for iter_idx in range(self.iters):
            iter_start = self._profile_start(f"Iter_{iter_idx}")
            
            N_minus_1 = self.N - 1
            if N_minus_1 == 0:
                # If N=1, we break. The final propagation happens after the loop.
                break

            C_total = N_minus_1 * self.M

            # --- OPTIMIZATION: Batched Propagation Strategy ---
            # We generate candidates first, then propagate N current trajectories 
            # and C candidates in a single, large batch.

            # 2a. Generate M Candidates (C_total)
            # Crucial Detail: Numba refines only N-1 trajectories (i=1 to N-1).
            decay_coeff = self.decay_coeffs[iter_idx]
            
            # Sample noise (C_total, T, nu) using chol_R (standard R used here) and apply decay
            noise_candidates = self._sample_noise_uge_to(C_total, self.T, self.chol_R) * decay_coeff
            
            # Repeat base action sequences (i=1 to N-1) M times
            action_seqs_repeated = action_seqs[1:].repeat_interleave(self.M, dim=0)
            
            cand_all_flat = action_seqs_repeated + noise_candidates
            cand_all_flat = self._clamp_controls(cand_all_flat)

            # 2b. Combine Batches (N + C_total)
            # OPTIMIZATION: Combine base (N) and candidate (C_total) actions.
            combined_actions = torch.cat([action_seqs, cand_all_flat], dim=0)

            # 2c. Propagate Combined Batch (N + C_total)
            # This replaces the two separate calls (Base/Candidate Propagation).
            prop_start = self._profile_start("CombinedPropagation")
            combined_trajs, combined_covs = self._propagate_mean_and_covariance(initial_state, combined_actions)
            self._profile_end("CombinedPropagation", prop_start)

            # 2d. Extract Gaussians at specific intervals (S)
            # Extract base trajectories (first N)
            means3d, covs3d = self._extract_gaussians_at_idx3d(
                combined_trajs[:self.N], combined_covs[:self.N], self.hellinger_indices
            )
            
            # Extract candidates (rest C_total)
            mu_c_all, Si_c_all = self._extract_gaussians_at_idx3d(
                combined_trajs[self.N:], combined_covs[self.N:], self.hellinger_indices
            )

            # 2e. Score and Select (Vectorized)
            # OPTIMIZATION: Use the fully vectorized JIT function instead of the Python loop.
            score_start = self._profile_start("HellingerScoring")
            
            # Call the optimized JIT kernel
            selected_actions_N_minus_1 = _score_and_select_vectorized_jit(
                mu_c_all, Si_c_all, means3d, covs3d, cand_all_flat,
                self.N, self.M, self.selection_mask
            )
            
            self._profile_end("HellingerScoring", score_start)
            
            # Update the action sequences (keeping index 0 unchanged)
            action_seqs[1:] = selected_actions_N_minus_1
            self._profile_end(f"Iter_{iter_idx}", iter_start)
            
        self._profile_end("UGE-TO_Iterations", iterations_start)

        # 3. Final Evaluation and Selection (Algorithm 2, Stage 2)
        final_start = self._profile_start("UGE-TO_Final")
        # Re-propagate the final diverse set. This is necessary for final cost evaluation 
        # and ensures consistency with the Numba logic, handling edge cases (iters=0, N=1).
        final_trajs, _ = self._propagate_mean_and_covariance(initial_state, action_seqs)

        # Calculate costs
        # Transpose final_trajs: (N, T+1, nx) -> (T+1, N, nx) for cost calculation
        final_trajs_transposed = final_trajs.permute(1, 0, 2)
        task_costs = self._calculate_trajectory_costs(final_trajs_transposed, goal_tensor)
        
        # Select the best trajectory (i*)
        best_idx = torch.argmin(task_costs).item()
        U_nominal_ugto = action_seqs[best_idx]
        self._profile_end("UGE-TO_Final", final_start)

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
        # 1. Propagate Mean (Standard rollout) - typically ~9ms
        mean_start = self._profile_start("MeanPropagation")
        # Transpose action_seqs: (B, T, nu) -> (T, B, nu)
        action_seqs_transposed = action_seqs.permute(1, 0, 2)
        # (T+1, B, nx)
        trajs_transposed = self._rollout_full_controls_torch(action_seqs_transposed, initial_state)
        # (B, T+1, nx)
        trajs = trajs_transposed.permute(1, 0, 2)
        self._profile_end("MeanPropagation", mean_start)

        # 2. Propagate Covariance (EKF style) - typically ~17ms
        cov_start = self._profile_start("CovariancePropagation")
        covs = self._propagate_covariance_pytorch(trajs, action_seqs)
        self._profile_end("CovariancePropagation", cov_start)
        
        return trajs, covs

    def _propagate_covariance_pytorch(self, trajs, actions_batch):
        """
        Wrapper for the highly optimized JIT EKF propagation.
        Prepares inputs (Jacobians and Noise terms) using vectorized operations.
        Update rule: Sigma_{t+1} = A_t Sigma_t A_t^T + B_t Q B_t^T
        """
        # Setup and tensor initialization (~0.5ms)
        B, T, _ = actions_batch.shape
        L = self.wheelbase
        dt = self.dt
        
        # Extract components for Jacobian calculation (Batched over B and T)
        V = actions_batch[:, :, 0]
        Delta = actions_batch[:, :, 1]
        Theta = trajs[:, :T, 2] # State at time t

        # Pre-calculate trigonometric functions
        SinTh = torch.sin(Theta)
        CosTh = torch.cos(Theta)
        TanDelta = torch.tan(Delta)

        eps = torch.tensor(1e-6, device=self.device, dtype=torch.float32)
        CosDeltaSq = torch.clamp(torch.cos(Delta)**2, min=eps)

        # --- Calculate Jacobian components (E = A-I and B) (Batched over B, T) ---
        # E components (A = I + E)
        a02 = -dt * V * SinTh
        a12 = dt * V * CosTh

        # B components
        b00 = dt * CosTh
        b10 = dt * SinTh
        b20 = dt * TanDelta / L
        b21 = dt * V / (L * CosDeltaSq)

        # Q components (Diagonal)
        q0 = self.Q_cov[0, 0]
        q1 = self.Q_cov[1, 1]

        # Pre-calculate BQB^T (B, T, nx, nx) using explicit expansion
        BQB = torch.zeros(B, T, self.nx, self.nx, device=self.device, dtype=torch.float32)
        
        # q0 terms
        BQB[:, :, 0, 0] = q0 * b00 * b00
        BQB[:, :, 0, 1] = q0 * b00 * b10
        BQB[:, :, 0, 2] = q0 * b00 * b20
        BQB[:, :, 1, 0] = BQB[:, :, 0, 1] # Symmetric
        BQB[:, :, 1, 1] = q0 * b10 * b10
        BQB[:, :, 1, 2] = q0 * b10 * b20
        BQB[:, :, 2, 0] = BQB[:, :, 0, 2] # Symmetric
        BQB[:, :, 2, 1] = BQB[:, :, 1, 2] # Symmetric
        BQB[:, :, 2, 2] = q0 * b20 * b20

        # q1 term
        BQB[:, :, 2, 2] += q1 * b21 * b21
        
        # Prepare initial Sigma batch (B, nx, nx)
        Sigma0_batch = self.Sigma0_cov.unsqueeze(0).repeat(B, 1, 1)

        # Transpose inputs for optimized JIT loop (B, T, ...) -> (T, B, ...)
        # .contiguous() ensures the memory layout is actually changed
        a02_T = a02.transpose(0, 1).contiguous()
        a12_T = a12.transpose(0, 1).contiguous()
        BQB_T = BQB.permute(1, 0, 2, 3).contiguous()

        # EKF Update Loop (JIT Compiled and Optimized)
        ekf_loop_start = self._profile_start("EKFLoop")
        # Call the optimized JIT function
        covs = _propagate_covariance_jit_optimized(
            T, B, self.nx, a02_T, a12_T, BQB_T, Sigma0_batch, self.device
        )
        self._profile_end("EKFLoop", ekf_loop_start)
        return covs

    def _extract_gaussians_at_idx3d(self, trajs, covs, idx):
        """Extracts means and covariances at specific time indices."""
        # Use torch.index_select along the time dimension (dim=1)
        means3d = torch.index_select(trajs, 1, idx) # (B, S, nx)
        covs3d = torch.index_select(covs, 1, idx)   # (B, S, nx, nx)
        return means3d, covs3d

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
        
        # Define the target visualization size based on the configuration (~0.1ms)
        TARGET_VIS_SIZE = self.num_vis_rollouts # e.g., 1000
        
        # Transform inputs to (Batch, T+1, nx) format and ensure contiguous memory layout
        uge_to_trajectories = uge_to_trajectories_T.permute(1, 0, 2).contiguous()
        mppi_trajectories = mppi_trajectories_T.permute(1, 0, 2).contiguous()
        N_total = uge_to_trajectories.shape[0]

        # --- 1. UGE-TO Best Sample - (Corresponds to 'cu_best' - Blue) --- (~0.1ms)
        # Shape: (T+1, nx)
        if N_total > 0:
            ugto_best_np = uge_to_trajectories[best_idx_ugto].cpu().numpy()
        else:
            # Handle edge case N=0 (should ideally not happen based on config checks)
            ugto_best_np = np.zeros((self.T + 1, self.nx), dtype=np.float32)

        # --- 2. UGE-TO Samples (Excluding Best) - (Corresponds to 'cu_samples' - Grey) --- (~0.6ms)
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
        # This is the most expensive visualization step (~11ms) due to rollout
        mppi_nominal_start = self._profile_start("VizMPPINominal")
        # Rollout the final nominal control sequence
        robot_frame_initial_state = torch.zeros(self.nx, device=self.device, dtype=torch.float32)
        U_nominal_reshaped = u_nominal_final.unsqueeze(1) # (T, 1, nu)
        
        # Use the refiner's rollout function
        nominal_traj_robot = self.mppi_refiner._rollout_full_controls_torch(U_nominal_reshaped, robot_frame_initial_state)
        
        # (T+1, 1, nx) -> (T+1, nx)
        mppi_nominal_np = nominal_traj_robot.squeeze(1).cpu().numpy()
        self._profile_end("VizMPPINominal", mppi_nominal_start)

        # --- 4. MPPI Samples - (Corresponds to 'mppi_samples' - Orange) --- (~0.3ms)
        # Initialize a full-sized array, pre-filled with the nominal trajectory (padding)
        mppi_samples_np = np.tile(mppi_nominal_np, (TARGET_VIS_SIZE, 1, 1))

        # Copy the actual MPPI samples
        num_actual_mppi_samples = mppi_trajectories.shape[0]
        num_to_copy_mppi = min(TARGET_VIS_SIZE, num_actual_mppi_samples)
        if num_to_copy_mppi > 0:
             # MPPI samples already exclude the nominal, so we copy them directly
             mppi_samples_np[:num_to_copy_mppi] = mppi_trajectories[:num_to_copy_mppi].cpu().numpy()

        # --- 5. Package Data --- (~2.3ms for angle wrapping and memory layout)
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