import numpy as np
import time
import numba
from numba import njit, prange
import gc
from uge_mpc_ros2.utils.vehicle import Vehicle
from uge_mpc_ros2.utils.input_constraints import convert_trajectories_to_costmap_indices_cpu
import math
import copy
import pickle
# =========================
# Config: use float32
# =========================
DTYPE = np.float32


# =========================
# Numba kernels (float32)
# =========================
@njit(parallel=True, fastmath=True)
def propagate_batch_numba(L, dt, x0, actions_batch):
    C, T, _ = actions_batch.shape
    trajs = np.zeros((C, T+1, 3), dtype=np.float32)
    for c in prange(C):
        x = x0[0]; y = x0[1]; th = x0[2]
        trajs[c, 0, 0] = x; trajs[c, 0, 1] = y; trajs[c, 0, 2] = th
        for t in range(T):
            v     = actions_batch[c, t, 0]
            delta = actions_batch[c, t, 1]
            x  = x  + dt * v * np.cos(th)
            y  = y  + dt * v * np.sin(th)
            th = th + dt * v / L * np.tan(delta)
            # normalize theta
            th = np.fmod(th, 2*np.pi)
            trajs[c, t+1, 0] = x
            trajs[c, t+1, 1] = y
            trajs[c, t+1, 2] = th      
    return trajs

njit(parallel=True, fastmath=True)
def propagate_batch_singletrack_numba(
    lf, lr, wheelbase, m, Iz, CSf, CSr, mu,
    dt, x0, actions_batch,
    v_min=0.0, v_max=5.0,
    delta_min=-np.pi/3, delta_max=np.pi/3
):
    """
    Batch propagate a single-track (bicycle) model with inputs v, delta.

    Dynamics (linear-tire small-angle form):
        beta_dot = mu/(v*(lf+lr)) * ( CSf*delta - (CSf+CSr)*beta + (CSr*lr - CSf*lf)*(r/v) ) - r
        r_dot    = mu*m/(Iz*(lf+lr)) * ( lf*CSf*delta + (CSr*lr - CSf*lf)*beta
                                         - (CSf*lf*lf + CSr*lr*lr)*(r/v) )
        psi_dot  = r
        x_dot    = v * cos(psi + beta)
        y_dot    = v * sin(psi + beta)

    Args:
        lf, lr : float   distances CoG->front/rear axle [m]
        m, Iz  : float   mass [kg], yaw inertia [kg m^2]
        CSf, CSr : float cornering coefficients [1/rad]
        mu     : float   tire-road friction coefficient
        dt     : float   step [s]
        x0     : (3,) or (5,) float32 initial state:
                 - if (3,): [x, y, theta], r=0, beta=0 assumed
                 - if (5,): [x, y, theta, r, beta]
        actions_batch : (C, T, 2) float32 with [v, delta]
        v_min, v_max : speed clamp [m/s]
        delta_min, delta_max : steering clamp [rad]

    Returns:
        trajs : (C, T+1, 3) float32 with [x, y, theta]
    """
    C, T, _ = actions_batch.shape
    trajs = np.zeros((C, T+1, 3), dtype=np.float32)

    wb = wheelbase
    two_pi = 2.0 * np.pi

    # Detect whether r, beta provided
    has_rb = (x0.size == 5)

    for c in prange(C):
        # init state
        x   = x0[0]
        y   = x0[1]
        psi = x0[2]
        if has_rb:
            r    = x0[3]
            beta = x0[4]
        else:
            r    = 0.0
            beta = 0.0

        trajs[c, 0, 0] = x
        trajs[c, 0, 1] = y
        trajs[c, 0, 2] = psi

        for t in range(T):
            v     = actions_batch[c, t, 0]
            delta = actions_batch[c, t, 1]

            # clamps
            if v < v_min: v = v_min
            elif v > v_max: v = v_max
            if delta < delta_min: delta = delta_min
            elif delta > delta_max: delta = delta_max

            v_safe = v if v > 1e-3 else 1e-3  # avoid division by ~0

            # single-track dynamics
            beta_dot = (
                mu / (v_safe * wb) *
                (CSf * delta - (CSf + CSr) * beta + (CSr * lr - CSf * lf) * (r / v_safe))
                - r
            )

            r_dot = (
                mu * m / (Iz * wb) *
                (lf * CSf * delta + (CSr * lr - CSf * lf) * beta
                 - (CSf * lf * lf + CSr * lr * lr) * (r / v_safe))
            )

            psi_dot = r
            x_dot   = v * np.cos(psi + beta)
            y_dot   = v * np.sin(psi + beta)

            # integrate (Euler)
            x   = x   + dt * x_dot
            y   = y   + dt * y_dot
            psi = psi + dt * psi_dot
            r   = r   + dt * r_dot
            beta= beta+ dt * beta_dot

            # normalize heading like your kinematic version
            psi = np.fmod(psi, two_pi)
            if psi < 0.0:
                psi += two_pi

            trajs[c, t+1, 0] = x
            trajs[c, t+1, 1] = y
            trajs[c, t+1, 2] = psi

    return trajs

@njit(parallel=True, fastmath=True, cache=True)
def propagate_uncertainty_batch_numba_fast(L, dt, trajs, actions_batch, Sigma0, Q):
    """
    Fast 3x3 covariance update:
      A = I + E with off-diagonals A[0,2]=a02, A[1,2]=a12
      BQB built from two columns b0, b1 with Q diag (q0, q1)
    """
    C, T, _ = actions_batch.shape
    covs = np.zeros((C, T+1, 3, 3), dtype=np.float32)
    q0 = Q[0, 0]
    q1 = Q[1, 1]

    # init
    for c in prange(C):
        covs[c, 0] = Sigma0
    # numba.cuda.syncthreads()
    Sigmas = np.zeros((C, 3, 3), dtype=np.float32)
    for c in prange(C):
        Sigmas[c] = Sigma0
    # numba.cuda.syncthreads()
    for c in prange(C):
        for t in range(T):
            v     = actions_batch[c, t, 0]
            delta = actions_batch[c, t, 1]
            th    = trajs[c, t, 2]

            a02 = -dt * v * np.sin(th)
            a12 =  dt * v * np.cos(th)

            b00 = dt * np.cos(th)
            b10 = dt * np.sin(th)
            b20 = dt * np.tan(delta) / L
            cosd = np.cos(delta)
            b21 = dt * v / (L * (cosd * cosd))

            S00 = Sigmas[c,0,0]; S01 = Sigmas[c,0,1]; S02 = Sigmas[c,0,2]
            S10 = Sigmas[c,1,0]; S11 = Sigmas[c,1,1]; S12 = Sigmas[c,1,2]
            S20 = Sigmas[c,2,0]; S21 = Sigmas[c,2,1]; S22 = Sigmas[c,2,2]

            # Start N = S
            N00 = S00; N01 = S01; N02 = S02
            N10 = S10; N11 = S11; N12 = S12
            N20 = S20; N21 = S21; N22 = S22

            # E S
            N00 += a02 * S20;  N01 += a02 * S21;  N02 += a02 * S22
            N10 += a12 * S20;  N11 += a12 * S21;  N12 += a12 * S22

            # S E^T
            N00 += a02 * S02;  N10 += a02 * S12;  N20 += a02 * S22
            N01 += a12 * S02;  N11 += a12 * S12;  N21 += a12 * S22

            # E S E^T (only top-left 2x2)
            add = S22
            N00 += a02 * a02 * add
            N01 += a02 * a12 * add
            N10 += a12 * a02 * add
            N11 += a12 * a12 * add

            # B Q B^T
            qb00 = q0 * b00 * b00; qb01 = q0 * b00 * b10; qb02 = q0 * b00 * b20
            qb10 = q0 * b10 * b00; qb11 = q0 * b10 * b10; qb12 = q0 * b10 * b20
            qb20 = q0 * b20 * b00; qb21 = q0 * b20 * b10; qb22 = q0 * b20 * b20
            N00 += qb00; N01 += qb01; N02 += qb02
            N10 += qb10; N11 += qb11; N12 += qb12
            N20 += qb20; N21 += qb21; N22 += qb22
            N22 += q1 * b21 * b21  # q1 term

            Sigmas[c,0,0]=N00; Sigmas[c,0,1]=N01; Sigmas[c,0,2]=N02
            Sigmas[c,1,0]=N10; Sigmas[c,1,1]=N11; Sigmas[c,1,2]=N12
            Sigmas[c,2,0]=N20; Sigmas[c,2,1]=N21; Sigmas[c,2,2]=N22

            covs[c, t+1, 0, 0] = N00; covs[c, t+1, 0, 1] = N01; covs[c, t+1, 0, 2] = N02
            covs[c, t+1, 1, 0] = N10; covs[c, t+1, 1, 1] = N11; covs[c, t+1, 1, 2] = N12
            covs[c, t+1, 2, 0] = N20; covs[c, t+1, 2, 1] = N21; covs[c, t+1, 2, 2] = N22

    return covs

@njit(parallel=True, fastmath=True)
def extract_gaussians_at_idx(trajs, covs, idx):
    C = trajs.shape[0]
    S = idx.shape[0]
    means2d = np.zeros((C, S, 2), dtype=np.float32)
    covs2d  = np.zeros((C, S, 2, 2), dtype=np.float32)
    for c in prange(C):
        for s_i in range(S):
            t = idx[s_i]
            means2d[c, s_i, 0] = trajs[c, t, 0]
            means2d[c, s_i, 1] = trajs[c, t, 1]
            covs2d[c, s_i, 0, 0] = covs[c, t, 0, 0]
            covs2d[c, s_i, 0, 1] = covs[c, t, 0, 1]
            covs2d[c, s_i, 1, 0] = covs[c, t, 1, 0]
            covs2d[c, s_i, 1, 1] = covs[c, t, 1, 1]
    return means2d, covs2d

@njit(parallel=True, fastmath=True)
def extract_gaussians_at_idx3d(trajs, covs, idx):
    C = trajs.shape[0]
    S = idx.shape[0]
    means3d = np.zeros((C, S, 3), dtype=np.float32)
    covs3d  = np.zeros((C, S, 3, 3), dtype=np.float32)
    for c in prange(C):
        for s_i in range(S):
            t = idx[s_i]
            means3d[c, s_i, 0] = trajs[c, t, 0]
            means3d[c, s_i, 1] = trajs[c, t, 1]
            means3d[c, s_i, 2] = trajs[c, t, 2]
            covs3d[c, s_i, 0, 0] = covs[c, t, 0, 0]
            covs3d[c, s_i, 0, 1] = covs[c, t, 0, 1]
            covs3d[c, s_i, 0, 2] = covs[c, t, 0, 2]
            covs3d[c, s_i, 1, 0] = covs[c, t, 1, 0]
            covs3d[c, s_i, 1, 1] = covs[c, t, 1, 1]
            covs3d[c, s_i, 1, 2] = covs[c, t, 1, 2]
            covs3d[c, s_i, 2, 0] = covs[c, t, 2, 0]
            covs3d[c, s_i, 2, 1] = covs[c, t, 2, 1]
            covs3d[c, s_i, 2, 2] = covs[c, t, 2, 2]
    return means3d, covs3d

@njit(fastmath=True)
def hellinger_3d_batch_numba_fast(mu_c, Sig_c, mu_o, Sig_o):
    """
    Fast 3D Hellinger^2 between batches of Gaussians over (x, y, theta).
    Formula:
      H^2 = 1 - ((|Σc|^{1/4} |Σo|^{1/4}) / |(Σc+Σo)/2|^{1/2})
                 * exp( -1/8 (μc-μo)^T ((Σc+Σo)/2)^{-1} (μc-μo) )
    Shapes:
      mu_c: (C, S, 3),  Sig_c: (C, S, 3, 3)
      mu_o: (K, S, 3),  Sig_o: (K, S, 3, 3)
    Returns:
      scores: (C,)
    """
    C, S, _ = mu_c.shape
    K = mu_o.shape[0]
    scores = np.zeros(C, dtype=np.float32)
    eps = DTYPE(1e-12)

    for ci in prange(C):
        ssum = DTYPE(0.0)
        for ki in range(K):
            for si in range(S):
                # mean diffs
                dx  = mu_c[ci, si, 0] - mu_o[ki, si, 0]
                dy  = mu_c[ci, si, 1] - mu_o[ki, si, 1]
                dth = mu_c[ci, si, 2] - mu_o[ki, si, 2]
                # If your theta is wrapped to [-pi, pi], uncomment the next line:
                # dth = (dth + DTYPE(np.pi))
                # dth = dth - DTYPE(2.0*np.pi) * DTYPE(math.floor(dth / (2.0*np.pi)))
                # dth = dth - DTYPE(np.pi)

                # Cov components (candidate)
                c00 = Sig_c[ci, si, 0, 0]; c01 = Sig_c[ci, si, 0, 1]; c02 = Sig_c[ci, si, 0, 2]
                c10 = Sig_c[ci, si, 1, 0]; c11 = Sig_c[ci, si, 1, 1]; c12 = Sig_c[ci, si, 1, 2]
                c20 = Sig_c[ci, si, 2, 0]; c21 = Sig_c[ci, si, 2, 1]; c22 = Sig_c[ci, si, 2, 2]
                # Cov components (other)
                o00 = Sig_o[ki, si, 0, 0]; o01 = Sig_o[ki, si, 0, 1]; o02 = Sig_o[ki, si, 0, 2]
                o10 = Sig_o[ki, si, 1, 0]; o11 = Sig_o[ki, si, 1, 1]; o12 = Sig_o[ki, si, 1, 2]
                o20 = Sig_o[ki, si, 2, 0]; o21 = Sig_o[ki, si, 2, 1]; o22 = Sig_o[ki, si, 2, 2]

                # A = 0.5 * (C + O)
                a00 = DTYPE(0.5) * (c00 + o00); a01 = DTYPE(0.5) * (c01 + o01); a02 = DTYPE(0.5) * (c02 + o02)
                a10 = DTYPE(0.5) * (c10 + o10); a11 = DTYPE(0.5) * (c11 + o11); a12 = DTYPE(0.5) * (c12 + o12)
                a20 = DTYPE(0.5) * (c20 + o20); a21 = DTYPE(0.5) * (c21 + o21); a22 = DTYPE(0.5) * (c22 + o22)

                # det(A)
                det_avg = (a00*(a11*a22 - a12*a21)
                          -a01*(a10*a22 - a12*a20)
                          +a02*(a10*a21 - a11*a20))
                if det_avg <= eps:
                    ssum += DTYPE(1.0)
                    continue

                # adj(A) entries (cofactors), then inv(A) = adj(A)^T / det(A)
                cof00 =  (a11*a22 - a12*a21)
                cof01 = -(a10*a22 - a12*a20)
                cof02 =  (a10*a21 - a11*a20)

                cof10 = -(a01*a22 - a02*a21)
                cof11 =  (a00*a22 - a02*a20)
                cof12 = -(a00*a21 - a01*a20)

                cof20 =  (a01*a12 - a02*a11)
                cof21 = -(a00*a12 - a02*a10)
                cof22 =  (a00*a11 - a01*a10)

                inv00 = cof00 / det_avg; inv01 = cof10 / det_avg; inv02 = cof20 / det_avg
                inv10 = cof01 / det_avg; inv11 = cof11 / det_avg; inv12 = cof21 / det_avg
                inv20 = cof02 / det_avg; inv21 = cof12 / det_avg; inv22 = cof22 / det_avg

                # quadratic form (μc-μo)^T A^{-1} (μc-μo)
                solx = inv00*dx + inv01*dy + inv02*dth
                soly = inv10*dx + inv11*dy + inv12*dth
                solz = inv20*dx + inv21*dy + inv22*dth
                quad = dx*solx + dy*soly + dth*solz

                exponent = DTYPE(-0.125) * quad

                # det(C), det(O)
                det_c = (c00*(c11*c22 - c12*c21)
                        -c01*(c10*c22 - c12*c20)
                        +c02*(c10*c21 - c11*c20))
                det_o = (o00*(o11*o22 - o12*o21)
                        -o01*(o10*o22 - o12*o20)
                        +o02*(o10*o21 - o11*o20))

                # clamp exp & pref for stability
                if exponent < DTYPE(-60.0):
                    inner = DTYPE(0.0)
                else:
                    # ensure non-neg in case of tiny roundoff
                    if det_c < eps or det_o < eps:
                        inner = DTYPE(0.0)
                    else:
                        pref = ((det_c * det_o) ** DTYPE(0.25)) / np.sqrt(det_avg)
                        inner = pref * np.exp(exponent)
                        if inner < DTYPE(0.0): inner = DTYPE(0.0)
                        if inner > DTYPE(1.0): inner = DTYPE(1.0)

                ssum += DTYPE(1.0) - inner
        scores[ci] = ssum
    return scores

@njit(fastmath=True)
def hellinger_2d_batch_numba_fast(mu_c, Sig_c, mu_o, Sig_o):
    """
    Fast 2x2 Hellinger: manual det/inv for float32.
    for x,y dimension
    """
    C, S, _ = mu_c.shape
    K = mu_o.shape[0]
    scores = np.zeros(C, dtype=np.float32)
    eps = DTYPE(1e-12)

    for ci in prange(C): # parallel over candidates
        ssum = DTYPE(0.0)
        for ki in range(K): # other
            for si in range(S): # step

                dx = mu_c[ci, si, 0] - mu_o[ki, si, 0]
                dy = mu_c[ci, si, 1] - mu_o[ki, si, 1]

                a = DTYPE(0.5) * (Sig_c[ci, si, 0, 0] + Sig_o[ki, si, 0, 0])
                b = DTYPE(0.5) * (Sig_c[ci, si, 0, 1] + Sig_o[ki, si, 0, 1])
                c = DTYPE(0.5) * (Sig_c[ci, si, 1, 0] + Sig_o[ki, si, 1, 0])
                d = DTYPE(0.5) * (Sig_c[ci, si, 1, 1] + Sig_o[ki, si, 1, 1])

                det_avg = a * d - b * c
                if det_avg <= eps:
                    ssum += DTYPE(1.0)
                    continue

                inv00 =  d / det_avg
                inv01 = -b / det_avg
                inv10 = -c / det_avg
                inv11 =  a / det_avg

                sol0 = inv00 * dx + inv01 * dy
                sol1 = inv10 * dx + inv11 * dy
                quad = dx * sol0 + dy * sol1
                exponent = DTYPE(-0.125) * quad

                ac = Sig_c[ci, si, 0, 0]; bc = Sig_c[ci, si, 0, 1]
                cc = Sig_c[ci, si, 1, 0]; dc = Sig_c[ci, si, 1, 1]
                ao = Sig_o[ki, si, 0, 0]; bo = Sig_o[ki, si, 0, 1]
                co = Sig_o[ki, si, 1, 0]; do = Sig_o[ki, si, 1, 1]
                det_c = ac * dc - bc * cc
                det_o = ao * do - bo * co

                # clamp exponent for stability
                if exponent < DTYPE(-60.0):
                    inner = DTYPE(0.0)
                else:
                    pref = ((det_c * det_o) ** 0.25) / np.sqrt(det_avg)
                    inner = pref * np.exp(exponent)
                    if inner < DTYPE(0.0): inner = DTYPE(0.0)
                    if inner > DTYPE(1.0): inner = DTYPE(1.0)

                ssum += DTYPE(1.0) - inner # 
        scores[ci] = ssum
    return scores

@njit(parallel=True, fastmath=True)
def score_and_select_fast(mu_c_all, Si_c_all, means2d, covs2d,
                          cand_all_flat, others_idx, num_trajs, candidates_per_traj):
    selected_actions = np.zeros((num_trajs-1, cand_all_flat.shape[1], 2), dtype=np.float32)
    for i in prange(1, num_trajs):
        start_idx = (i-1) * candidates_per_traj
        end_idx   = start_idx + candidates_per_traj
        mu_c = mu_c_all[start_idx:end_idx]
        Si_c = Si_c_all[start_idx:end_idx]
        mu_o = means2d[others_idx[i]]
        Si_o = covs2d[others_idx[i]]
            # add the added noise to the base actions
        hellinger_scores = hellinger_2d_batch_numba_fast(mu_c, Si_c, mu_o, Si_o)
        # argmax manual (Numba-friendly)
        # add the cost functions for the candidates
        # find the best index of the scores
        best_idx = 0
        best     = hellinger_scores[0]
        for k in range(1, hellinger_scores.shape[0]):
            if hellinger_scores[k] > best:
                best = hellinger_scores[k]
                best_idx = k
        selected_actions[i-1] = cand_all_flat[start_idx + best_idx]
    return selected_actions

@njit(parallel=True, fastmath=True)
def score_and_select_fast_3d(mu_c_all, Si_c_all, means3d, covs3d,
                             cand_all_flat, others_idx, num_trajs, candidates_per_traj):
    selected_actions = np.zeros((num_trajs-1, cand_all_flat.shape[1], 2), dtype=np.float32)
    for i in prange(1, num_trajs):
        start_idx = (i-1) * candidates_per_traj
        end_idx   = start_idx + candidates_per_traj
        mu_c = mu_c_all[start_idx:end_idx]
        Si_c = Si_c_all[start_idx:end_idx]
        mu_o = means3d[others_idx[i]]
        Si_o = covs3d[others_idx[i]]

        hellinger_scores = hellinger_3d_batch_numba_fast(mu_c, Si_c, mu_o, Si_o)

        best_idx = 0
        best     = hellinger_scores[0]
        for k in range(1, hellinger_scores.shape[0]):
            if hellinger_scores[k] > best:
                best = hellinger_scores[k]
                best_idx = k
        selected_actions[i-1] = cand_all_flat[start_idx + best_idx]
    return selected_actions

# write a function that takes the trajectory and the goal and returns whether the trajectory reaches the goal or not 
@njit(fastmath=True)
def check_goal_reached(traj, goal_x, goal_y, goal_theta, goal_tolerance=0.2, angle_tolerance=0.17):
    '''
    Find the ones that reaches the goal and the ones that don't
    return an array that show the index of the first time step that the trajectory reaches the goal
    if the trajectory doesn't reach the goal, return the last time step

    '''
    T, _ = traj.shape
    goal_reached = -1 # to make it -1 if the trajectory doesn't reach the goal
    for i in prange(T):
        xy_dist_check = np.sqrt((traj[i, 0] - goal_x)**2 + (traj[i, 1] - goal_y)**2) < goal_tolerance
        angle_dist_check = np.sqrt(((np.cos(goal_theta) - np.cos(traj[i, 2]))**2 
                    + (np.sin(goal_theta) - np.sin(traj[i, 2]))**2)) < angle_tolerance
        if xy_dist_check==True and angle_dist_check==True:
            goal_reached = i
            break
    if goal_reached == -1:
        goal_reached = T-1 # to make it the last time step if the trajectory doesn't reach the goal
    return goal_reached

@njit(fastmath=True)
def action_smoothness_batch(action_seq, T):
    """
    action_seq: (T, U) -> per-trajectory smoothness cost
    """
    U = action_seq.shape[1] # number of actions
    diff_sum = np.float32(0.0)
    for t in range(1, T):
        for u in range(U): # number of actions
            d = action_seq[t, u] - action_seq[t-1, u]
            diff_sum += d * d
    return diff_sum

@njit(parallel=True, fastmath=True, cache=True)
def compute_cost(trajs, goal_x, goal_y, goal_theta, circles, action_seqs, discount_factor=1.0, theta_weight=1.0, dist_weight=1.5, terminal_weight=10.0, action_weight=1.0, obstacle_weight=1.0):
    """
    trajs: (N, T, 3) float32 array
    goal_x, goal_y, goal_theta: float32
    discount_factor: float32, discount factor for future timesteps (0 < discount_factor <= 1)
    returns: (N,) array with discounted distance-to-goal for each trajectory
    """
    N, TT, _ = trajs.shape # N: number of trajectories, T: number of timesteps + 1 (60 + 1)
    dist2 = np.empty(N, dtype=np.float32)
    dist_cost_return = np.empty(N, dtype=np.float32)
    theta_cost_return = np.empty(N, dtype=np.float32)
    terminal_cost_return = np.empty(N, dtype=np.float32)
    action_cost_return = np.empty(N, dtype=np.float32)
    num_goal_reached = np.zeros(N, dtype=np.int32)
    goal_reached_idx = np.empty(N, dtype=np.int32)
    obstacle_cost_return = np.empty(N, dtype=np.float32)
    # Pre-compute discount factors for all timesteps
    discount_factors = np.empty(TT, dtype=np.float32)
    is_collided = np.zeros(TT, dtype=np.int32)
    for t in range(TT):
        discount_factors[t] = discount_factor ** t

    for i in prange(N):
        # take every state and then find the min distance
        x_curr = trajs[i, :, 0]
        y_curr = trajs[i, :, 1]
        theta_curr = trajs[i, :, 2]
        dx = goal_x - x_curr
        dy = goal_y - y_curr
        distances = np.sqrt(dx * dx + dy * dy)
        theta_costs = np.sqrt((np.cos(goal_theta) - np.cos(theta_curr))**2 
                            + (np.sin(goal_theta) - np.sin(theta_curr))**2).astype(np.float32)
        # Apply pre-computed discount factors and use minimum (consistent with original 3D version)
        # dist2[i] = np.min(distances)
        # add the cost function for the trajectory
        goal_reached = check_goal_reached(trajs[i], goal_x, goal_y, goal_theta, goal_tolerance=0.4, angle_tolerance=0.17) 
        goal_reached_idx[i] = goal_reached
        # count number of trajectories that reach the goal
         # to make it 0 if the trajectory doesn't reach the goal

        # add the obstacle cost 
        # obstacle cost is the exp(-(x from the circle center)^2 - (y from the circle center)^2 / (2 * radius^2))
        # calculate the distance from the circle center
        # for each circle, calculate the distance from the circle center
        obstacles_cost = np.zeros(TT, dtype=np.float32)
        for j in range(circles.shape[0]):
            dx_circle = x_curr - circles[j, 0]
            dy_circle = y_curr - circles[j, 1]
            dist_circle = np.sqrt(dx_circle**2 + dy_circle**2)
            obstacles_cost += np.exp(-dist_circle**2 / (2 * circles[j, 2]**2)).astype(np.float32)


        if goal_reached == TT-1:
            # print('goal not reached')
            dist_cost = np.sum(distances * discount_factors)
            dist_cost_return[i] = dist_weight * dist_cost
            # add the cost function for the theta
            theta_cost = np.sum(theta_costs * discount_factors)
            theta_cost_return[i] = theta_weight * theta_cost
            num_goal_reached[i] = 0
            # terminal_cost = np.min(distances)
            terminal_cost = distances[TT-1]
            terminal_cost_return[i] = terminal_weight * terminal_cost
            # action_cost should be the sum of the consecutive actions difference
            action_cost = action_smoothness_batch(action_seqs[i], TT-2) # TT-2 because the action sequence is (T-1, 2)
            action_cost_return[i] = action_weight * action_cost

            obstacle_cost_return[i] = np.sum(obstacles_cost)
        else: # if the trajectory reaches the goal
            # print('goal_reached at', goal_reached)
            dist_cost = np.sum(distances[:goal_reached] * discount_factors[:goal_reached])
            dist_cost_return[i] = dist_weight * dist_cost
            theta_cost = np.sum(theta_costs[:goal_reached] * discount_factors[:goal_reached])
            theta_cost_return[i] = theta_weight * theta_cost
            num_goal_reached[i] = 1
            # action_cost sum until the goal is reached
            action_cost = action_smoothness_batch(action_seqs[i], goal_reached)
            action_cost_return[i] = action_weight * action_cost

            obstacle_cost_return[i] = np.sum(obstacles_cost[:goal_reached])
            # add terminal cost for the min distance from the goal of the trajectory
            terminal_cost = distances[goal_reached]
            terminal_cost_return[i] = terminal_weight * terminal_cost 
        # Ensure all values are float32 before assignment     
        total_cost = (dist_weight * dist_cost +  
                     terminal_weight * terminal_cost +
                     action_weight * action_cost) # +
                    #  obstacle_weight * obstacle_cost_return[i])
        # total_cost = (dist_weight * dist_cost + 
        #              theta_weight * theta_cost + 
        #              terminal_weight * terminal_cost +
        #              action_weight * action_cost)
        dist2[i] = total_cost
        
    return dist2, dist_cost_return, theta_cost_return, terminal_cost_return, action_cost_return, np.sum(num_goal_reached), num_goal_reached, goal_reached_idx


# =========================
# UAE_method class wrapper
# =========================
class UAE_method:
    """
    Lightweight class that organizes the UAE kernels and optimizer.
    Kernels remain top-level @njit functions for best Numba performance.
    """
    def __init__(self, vehicle, dtype=DTYPE, discount_factor=0.95, dist_weight=4.0, theta_weight=1.0, terminal_weight=20.0, action_weight=1.0, obstacle_weight=1.0):
        self.vehicle = vehicle
        self.dtype = dtype
        self.discount_factor = np.float32(discount_factor)
        self.curr_x = None
        self.curr_u = None
        self.xgoal = np.array([10.0, 10.0, 0.0]) # dummy goal
        self.goal_tolerance = 0.2 # dummy goal tolerance
        self.theta_weight = theta_weight
        self.dist_weight = dist_weight
        self.terminal_weight = terminal_weight
        self.action_weight = action_weight
        self.obstacle_weight = obstacle_weight
        # bind kernel refs (optional, but keeps call sites tidy)
        if vehicle.vehicle_type == "singletrack":
            # self.propagate_kernel = propagate_batch_singletrack_numba
            self.propagate_kernel = propagate_batch_numba
        elif vehicle.vehicle_type == "kinematic":
            self.propagate_kernel = propagate_batch_numba
        
        self.cov_kernel = propagate_uncertainty_batch_numba_fast
        self.extract_kernel = extract_gaussians_at_idx
        self.hellinger_kernel = hellinger_2d_batch_numba_fast
        self.select_kernel = score_and_select_fast
        self.extract_kernel3d = extract_gaussians_at_idx3d
        self.hellinger_kernel3d = hellinger_3d_batch_numba_fast
        self.select_kernel3d = score_and_select_fast_3d
        self.check_goal_reached = check_goal_reached
        self.compute_cost_uae = compute_cost

   
    def optimize(self, x0, base_actions, R, Sigma0, Q,
                 num_trajs=10, iters=10, candidates_per_traj=5, step_interval=10,
                 v_bounds=(0.5, 2.0), delta_bounds=(-np.pi/6, np.pi/6),
                 random_seed=42, decay_sharpness=1.0):
        """
        Orchestrates trajectory optimization using the compiled kernels.
        Final actions and the trajectories are returned for the cost calculation
        Returns: action_seqs, final_trajs
        """

        rng = np.random.default_rng(random_seed)
        # cast inputs to float32
        x0 = x0.astype(self.dtype)
        base_actions = base_actions.astype(self.dtype)
        R = R.astype(self.dtype); Sigma0 = Sigma0.astype(self.dtype); Q = Q.astype(self.dtype)

        action_seqs = [base_actions] + [
            (base_actions + rng.multivariate_normal(np.zeros(2, dtype=np.float64),
                                                    R.astype(np.float64),
                                                    size=base_actions.shape[0]).astype(self.dtype))
            for _ in range(num_trajs - 1)
        ] # (N,T,2)
        final_trajs = np.zeros((num_trajs, base_actions.shape[0]+1, 3), dtype=self.dtype)
        
        L, dt = self.vehicle.wheelbase, self.vehicle.dt
        T = base_actions.shape[0]
        idx = np.arange(0, T+1, step_interval, dtype=np.int64)

        # Cholesky for R (float32 safe)
        chol_R = np.linalg.cholesky(R.astype(np.float64)).astype(self.dtype)

        # others indices
        others_idx = [np.array([j for j in range(num_trajs) if j != i], dtype=np.int64)
                      for i in range(num_trajs)]
        # Exponential decay schedule (2.0 -> 1.0 across iters)
        decay_coeffs = np.exp(np.linspace(np.log(2.0), np.log(1.0), iters) ** decay_sharpness).astype(self.dtype)
    
        for iter_idx in range(iters):
            t0 = time.perf_counter()

            curr_actions = np.stack(action_seqs).astype(self.dtype)                 # (N,T,2)
            trajs = self.propagate_kernel(L, dt, x0, curr_actions)                  # (N,T+1,3)
            covs  = self.cov_kernel(L, dt, trajs, curr_actions, Sigma0, Q)
            means2d, covs2d = self.extract_kernel(trajs, covs, idx)

            # ---- candidates batch ----
            Z = rng.standard_normal(size=((num_trajs-1), candidates_per_traj, T, 2)).astype(self.dtype)
            # noise = Z @ chol_R^T
            noise_all = np.einsum('nctk,lk->nctl', Z, chol_R.T, dtype=self.dtype)

            # Apply exponentially decaying coefficient to noise
            noise_all = noise_all * decay_coeffs[iter_idx]
            # print(f"Decay coefficient: {decay_coeffs[iter_idx]}")

            cand_all = curr_actions[1:, None, :, :] + noise_all
            # clip
            cand_all[:, :, :, 0] = np.clip(cand_all[:, :, :, 0], self.dtype(v_bounds[0]), self.dtype(v_bounds[1]))
            cand_all[:, :, :, 1] = np.clip(cand_all[:, :, :, 1], self.dtype(delta_bounds[0]), self.dtype(delta_bounds[1]))
            cand_all_flat = cand_all.reshape((num_trajs-1)*candidates_per_traj, T, 2)

            # propagate candidates
            trajs_c_all = self.propagate_kernel(L, dt, x0, cand_all_flat)
            covs_c_all  = self.cov_kernel(L, dt, trajs_c_all, cand_all_flat, Sigma0, Q)
            mu_c_all, Si_c_all = self.extract_kernel(trajs_c_all, covs_c_all, idx)

            # parallel scoring & selection
            selected_actions = self.select_kernel(mu_c_all, Si_c_all, means2d, covs2d,
                                                  cand_all_flat, others_idx, num_trajs, candidates_per_traj)

            action_seqs = [action_seqs[0]] + [selected_actions[k] for k in range(num_trajs-1)]

        final_trajs = self.propagate_kernel(L, dt, x0, np.stack(action_seqs).astype(self.dtype))
        return action_seqs, final_trajs
    
    def optimize3D(self, x0, base_actions, R, Sigma0, Q,
                 num_trajs=10, iters=10, candidates_per_traj=5, step_interval=10,
                 v_bounds=(0.5, 2.0), delta_bounds=(-np.pi/6, np.pi/6),
                 random_seed=42, decay_sharpness=1.0):
        """
        Orchestrates trajectory optimization using the compiled kernels.
        Final actions and the trajectories are returned for the cost calculation
        Returns: action_seqs, final_trajs
        """

        rng = np.random.default_rng(random_seed)
        # cast inputs to float32
        x0 = x0.astype(self.dtype)
        base_actions = base_actions.astype(self.dtype)
        R = R.astype(self.dtype); Sigma0 = Sigma0.astype(self.dtype); Q = Q.astype(self.dtype)

        action_seqs = [base_actions] + [
            (base_actions + rng.multivariate_normal(np.zeros(2, dtype=np.float64),
                                                    3*R.astype(np.float64),
                                                    size=base_actions.shape[0]).astype(self.dtype))
            for _ in range(num_trajs - 1)
        ] # (N,T,2)
        
        final_trajs = np.zeros((num_trajs, base_actions.shape[0]+1, 3), dtype=self.dtype)
        
        L, dt = self.vehicle.wheelbase, self.vehicle.dt
        T = base_actions.shape[0]
        idx = np.arange(0, T+1, step_interval, dtype=np.int64)

        # Cholesky for R (float32 safe)
        chol_R = np.linalg.cholesky(R.astype(np.float64)).astype(self.dtype)

        # others indices
        others_idx = [np.array([j for j in range(num_trajs) if j != i], dtype=np.int64)
                      for i in range(num_trajs)]
        
        # Exponential decay schedule (2.0 -> 1.0 across iters)
        decay_coeffs = np.exp(np.linspace(np.log(2.0), np.log(1.0), iters) ** decay_sharpness).astype(self.dtype)
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled: 
            gc.disable()

        for iter_idx in range(iters):
            
            curr_actions = np.stack(action_seqs).astype(self.dtype) # (N,T,2)
            
            trajs = self.propagate_kernel(L, dt, x0, curr_actions)                  # (N,T+1,3)
            covs  = self.cov_kernel(L, dt, trajs, curr_actions, Sigma0, Q)
            # NEW (3D):
            means3d, covs3d = self.extract_kernel3d(trajs, covs, idx)
            
            
            
            # ---- candidates batch ----
            Z = rng.standard_normal(size=((num_trajs-1), candidates_per_traj, T, 2)).astype(self.dtype)
            # noise = Z @ chol_R^T
            noise_all = np.einsum('nctk,lk->nctl', Z, chol_R.T, dtype=self.dtype)

            # Apply exponentially decaying coefficient to noise
            noise_all = noise_all * decay_coeffs[iter_idx]
            # noise_all = noise_all * 1.0


            cand_all = curr_actions[1:, None, :, :] + noise_all
            # clip
            cand_all[:, :, :, 0] = np.clip(cand_all[:, :, :, 0], self.dtype(v_bounds[0]), self.dtype(v_bounds[1]))
            cand_all[:, :, :, 1] = np.clip(cand_all[:, :, :, 1], self.dtype(delta_bounds[0]), self.dtype(delta_bounds[1]))
            cand_all_flat = cand_all.reshape((num_trajs-1)*candidates_per_traj, T, 2)

            # propagate candidates
            trajs_c_all = self.propagate_kernel(L, dt, x0, cand_all_flat)
            covs_c_all  = self.cov_kernel(L, dt, trajs_c_all, cand_all_flat, Sigma0, Q)
            
            mu_c_all, Si_c_all = self.extract_kernel3d(trajs_c_all, covs_c_all, idx)
            
            selected_actions = self.select_kernel3d(mu_c_all, Si_c_all, means3d, covs3d,
                                                    cand_all_flat, others_idx, num_trajs, candidates_per_traj)
            action_seqs = [action_seqs[0]] + [selected_actions[k] for k in range(num_trajs-1)]
            # print('time taken to select: ', time.perf_counter() - t1)
        
        if gc_was_enabled: gc.enable()
        final_trajs = self.propagate_kernel(L, dt, x0, np.stack(action_seqs).astype(self.dtype))
        return action_seqs, final_trajs

    def optimize3D_plot(self, x0, base_actions, R, Sigma0, Q,
                 num_trajs=10, iters=10, candidates_per_traj=5, step_interval=10,
                 v_bounds=(0.5, 2.0), delta_bounds=(-np.pi/6, np.pi/6),
                 random_seed=42, decay_sharpness=1.0):
        """
        Orchestrates trajectory optimization using the compiled kernels.
        Returns: history_means, history_covs, iter_times
        """
        rng = np.random.default_rng(random_seed)

        # cast inputs to float32
        x0 = x0.astype(self.dtype)
        base_actions = base_actions.astype(self.dtype)
        R = R.astype(self.dtype); Sigma0 = Sigma0.astype(self.dtype); Q = Q.astype(self.dtype)

        action_seqs = [base_actions] + [
            (base_actions + rng.multivariate_normal(np.zeros(2, dtype=np.float64),
                                                    R.astype(np.float64),
                                                    size=base_actions.shape[0]).astype(self.dtype))
            for _ in range(num_trajs - 1)
        ]

        iter_times = []
        history_means, history_covs = [], []
        L, dt = self.vehicle.wheelbase, self.vehicle.dt

        T = base_actions.shape[0]
        idx = np.arange(0, T+1, step_interval, dtype=np.int64)

        # Cholesky for R (float32 safe)
        chol_R = np.linalg.cholesky(R.astype(np.float64)).astype(self.dtype)

        # others indices
        others_idx = [np.array([j for j in range(num_trajs) if j != i], dtype=np.int64)
                      for i in range(num_trajs)]

        # Exponential decay schedule (2.0 -> 1.0 across iters)
        decay_coeffs = np.exp(np.linspace(np.log(1.0), np.log(1.0), iters) ** decay_sharpness).astype(self.dtype)

        for iter_idx in range(iters):
            t0 = time.perf_counter()

            curr_actions = np.stack(action_seqs).astype(self.dtype)                 # (N,T,2)
            trajs = self.propagate_kernel(L, dt, x0, curr_actions)                  # (N,T+1,3)
            covs  = self.cov_kernel(L, dt, trajs, curr_actions, Sigma0, Q)
            # means2d, covs2d = self.extract_kernel(trajs, covs, idx)
            # NEW (3D):
            means3d, covs3d = self.extract_kernel3d(trajs, covs, idx)

            history_means.append(trajs.copy())
            history_covs.append(covs.copy())

            # ---- candidates batch ----
            Z = rng.standard_normal(size=((num_trajs-1), candidates_per_traj, T, 2)).astype(self.dtype)
            # noise = Z @ chol_R^T
            noise_all = np.einsum('nctk,lk->nctl', Z, chol_R.T, dtype=self.dtype)

            # Apply exponentially decaying coefficient to noise
            noise_all = noise_all * decay_coeffs[iter_idx]
            # print(f"Decay coefficient: {decay_coeffs[iter_idx]}")

            cand_all = curr_actions[1:, None, :, :] + noise_all
            # clip
            cand_all[:, :, :, 0] = np.clip(cand_all[:, :, :, 0], self.dtype(v_bounds[0]), self.dtype(v_bounds[1]))
            cand_all[:, :, :, 1] = np.clip(cand_all[:, :, :, 1], self.dtype(delta_bounds[0]), self.dtype(delta_bounds[1]))
            cand_all_flat = cand_all.reshape((num_trajs-1)*candidates_per_traj, T, 2)

            # propagate candidates
            trajs_c_all = self.propagate_kernel(L, dt, x0, cand_all_flat)
            covs_c_all  = self.cov_kernel(L, dt, trajs_c_all, cand_all_flat, Sigma0, Q)
            mu_c_all, Si_c_all = self.extract_kernel3d(trajs_c_all, covs_c_all, idx)

            # parallel scoring & selection
            selected_actions = self.select_kernel3d(mu_c_all, Si_c_all, means3d, covs3d,
                                                    cand_all_flat, others_idx, num_trajs, candidates_per_traj)

            action_seqs = [action_seqs[0]] + [selected_actions[k] for k in range(num_trajs-1)]
            iter_times.append(time.perf_counter() - t0)
        # also return the final actions for the cost calculation
        return history_means, history_covs, iter_times, action_seqs

    def solve(self, action_seqs, final_trajs, circles):
        """
        Solve the problem using the UAE method
        """
        
        # convert the final_trajs to the grid coordinates
        # grid_resolution = 0.05
        # grid_origin = [-0.5, -2.25]
        # final_trajs_xy = np.ascontiguousarray(final_trajs[:, :, :2], dtype=np.float32)
        
        # inv_res = np.float32(1.0 / 0.05)
        # origin_x = np.float32(-0.5)
        # origin_y = np.float32(-2.25)

        # gx, gy = convert_trajectories_to_costmap_indices_cpu(final_trajs_xy, inv_res, origin_x, origin_y)
        # final_trajs_grid_x, final_trajs_grid_y = convert_trajectories_to_costmap_indices_cpu(final_trajs_xy, 0.05, origin=[-0.5, -2.25]) 

        # distance to goal cost
        goal_x, goal_y = np.float32(self.xgoal[0]), np.float32(self.xgoal[1])
        goal_theta = np.float32(self.xgoal[2])
        # print(f"goal_x: {goal_x}, goal_y: {goal_y}, goal_theta: {goal_theta}")
        final_trajs_cont = np.ascontiguousarray(final_trajs[:, :, :], dtype=np.float32)
        action_seqs_cont = np.ascontiguousarray(action_seqs, dtype=np.float32)
        # compute the squared distance to goal
        total_cost,dist_cost,theta_cost,terminal_cost, \
            action_cost, num_goal_reached, \
                num_goal_reached_idx, goal_reached_idx = compute_cost(final_trajs_cont, goal_x, goal_y, 
                                                                      goal_theta, circles, action_seqs_cont, 
                                                                      self.discount_factor, self.theta_weight, 
                                                                      self.dist_weight, self.terminal_weight, 
                                                                      self.action_weight)

        print(f'dist_cost: {dist_cost}')
        # select the min cost trajectory
        min_cost_idx = np.argmin(total_cost)

        useq = np.zeros((np.array(action_seqs).shape[1], 2), dtype=self.dtype)
        # if the first trajectory from final_trajs_contreaches the goal, then use the first trajectory
        if num_goal_reached_idx[0] == 1:
            useq = action_seqs[0][:]
            # useq[goal_reached_idx[0]:] = 0.0
        else:
            useq = action_seqs[min_cost_idx][:]


        return useq, min_cost_idx, total_cost

    def solve_mppi(self, action_seqs, final_trajs, base_actions, circles, lambda_weight=0.5, v_bounds=(0.1, 2.0), delta_bounds=(-np.pi/6, np.pi/6)):
        """
        Solve the problem using the MPPI method with weighted averaging of control sequences
        
        Args:
            action_seqs: List of action sequences, each of shape (T, 2)
            final_trajs: Array of final trajectories of shape (N, T+1, 3)
            lambda_weight: Temperature parameter for MPPI weighting (default: 1.0)
            v_bounds: Tuple of (v_min, v_max) for velocity bounds (default: (0.5, 2.0))
            delta_bounds: Tuple of (delta_min, delta_max) for steering bounds (default: (-pi/6, pi/6))
            
        Returns:
            useq: Updated control sequence of shape (T, 2)
            min_cost_idx: Index of the minimum cost trajectory
            dist_to_goal2: Array of distances to goal for all trajectories
        """
        # Convert final trajectories to xy coordinates
        # distance to goal cost
        goal_x, goal_y = np.float32(self.xgoal[0]), np.float32(self.xgoal[1])
        goal_theta = np.float32(self.xgoal[2])
        # print(f"goal_x: {goal_x}, goal_y: {goal_y}, goal_theta: {goal_theta}")
        final_trajs_cont = np.ascontiguousarray(final_trajs[:, :, :], dtype=np.float32)
        action_seqs_cont = np.ascontiguousarray(action_seqs, dtype=np.float32)
        
        # compute the squared distance to goal
        total_cost,dist_cost,theta_cost,terminal_cost, action_cost, num_goal_reached, num_goal_reached_idx, goal_reached_idx = compute_cost(final_trajs_cont, goal_x, goal_y, goal_theta, circles, action_seqs_cont, self.discount_factor, self.theta_weight, self.dist_weight, self.terminal_weight, self.action_weight)
        # print(f"total cost mppi: {total_cost}")
        # print(f"num_goal_reached_mppi: {num_goal_reached}")
        # print(f"dist_to_goal2: {dist_to_goal2}")
        
        # Find the minimum cost trajectory index
        min_cost_idx = np.argmin(total_cost)
        
        # Get the number of trajectories and timesteps
        num_trajs = len(action_seqs)
        T = action_seqs[0].shape[0]
        
        # Convert action_seqs to numpy array for easier manipulation
        action_seqs_array = np.stack(action_seqs).astype(self.dtype)  # Shape: (N, T, 2)
        
        # MPPI update logic (adapted from mppi_barn.py)
        # Find the minimum cost (beta) for numerical stability
        beta = np.min(total_cost)
        
        # Compute weights using exponential cost transformation
        weights = np.exp(-(1.0 / lambda_weight) * (total_cost - beta))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Initialize the updated control sequence
        useq = action_seqs_array[0].copy() # base action sequence
        # useq = base_actions.copy()
        
        # Weighted average of control sequences
        for i in range(num_trajs):
            useq += weights[i] * (action_seqs_array[i] - base_actions)
        
        # Apply control bounds (clipping)
        useq[:, 0] = np.clip(useq[:, 0], v_bounds[0], v_bounds[1])      # Velocity bounds
        useq[:, 1] = np.clip(useq[:, 1], delta_bounds[0], delta_bounds[1])  # Steering bounds
        
        return useq, min_cost_idx, total_cost, goal_reached_idx

    def solve_mppi_only_with_total_cost(self, total_cost, action_seqs, base_actions, lambda_weight=0.5, v_bounds=(0.1, 2.0), delta_bounds=(-np.pi/6, np.pi/6)):
        """
        Solve the problem using the MPPI method with weighted averaging of control sequences        
        """

        total_cost_copied = copy.deepcopy(total_cost)
        # MPPI update logic (adapted from mppi_barn.py)
        # Find the minimum cost (beta) for numerical stability
        beta = np.min(total_cost)
        min_cost_idx = np.argmin(total_cost)
        # Compute weights using exponential cost transformation
        weights = np.exp(-(1.0 / lambda_weight) * (total_cost - beta))
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Initialize the updated control sequence
        useq = copy.deepcopy(action_seqs[0]) # base action sequence
        # useq = base_actions.copy()
        num_trajs = len(total_cost)
        # Weighted average of control sequences
        for i in range(num_trajs):
            useq += weights[i] * (action_seqs[i] - base_actions)
        
        # Apply control bounds (clipping)
        useq[:, 0] = np.clip(useq[:, 0], v_bounds[0], v_bounds[1])      # Velocity bounds
        useq[:, 1] = np.clip(useq[:, 1], delta_bounds[0], delta_bounds[1])  # Steering bounds
        
        return useq, min_cost_idx

    def generate_rollouts(self, useq, T, x0=None, num_rollouts=500, noise_covariance=None, include_base=True, v_bounds=(0.1, 4.0), delta_bounds=(-np.pi/6, np.pi/6)):
        """
        Generate rollout states from a control sequence for visualization.
        
        Args:
            useq: Control sequence of shape (T, 2) with [v, delta] for each timestep
            T: Time horizon
            x0: Initial state [x, y, theta]. If None, uses origin [0, 0, 0]
            num_rollouts: Number of rollouts to generate (excluding base if include_base=True)
            noise_std: Standard deviation of noise to add to controls for diversity
            include_base: If True, includes the base trajectory (no noise) as the first rollout
            
        Returns:
            rollout_states: Array of shape (num_rollouts + include_base, T+1, 3) containing [x, y, theta] for each rollout
        """
        # Input validation
        if useq.shape[0] != T:
            raise ValueError(f"Control sequence length {useq.shape[0]} doesn't match time horizon T={T}")
        
        if useq.shape[1] != 2:
            raise ValueError(f"Control sequence should have 2 dimensions [v, delta], got {useq.shape[1]}")
        
        # Ensure inputs are float32
        useq = useq.astype(self.dtype)
        
        # Set default initial state if not provided
        if x0 is None:
            x0 = np.array([0.0, 0.0, 0.0], dtype=self.dtype)
        else:
            x0 = x0.astype(self.dtype)
        
        if x0.shape[0] != 3:
            raise ValueError(f"Initial state should have 3 dimensions [x, y, theta], got {x0.shape[0]}")
        
        # Create multiple control sequences by adding noise
        L, dt = self.vehicle.wheelbase, self.vehicle.dt
        
        if include_base:
            # Include base trajectory as first rollout
            total_rollouts = num_rollouts + 1
            actions_list = [useq]  # Base trajectory first
        else:
            total_rollouts = num_rollouts
            actions_list = []
        
        if noise_covariance is not None:
            # Generate noise from multivariate normal with covariance R
            # rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            noise = np.random.multivariate_normal(np.zeros(2), noise_covariance, size=(num_rollouts, T)).astype(self.dtype)
            # Add noise to the base control sequence
            noisy_actions = useq + noise  # Shape: (num_rollouts, T, 2)
        else:
            assert False, "Noise covariance must be provided"
        
        # Stack all actions
        if include_base:
            # Reshape base actions to match noisy_actions shape for concatenation
            base_actions = actions_list[0].reshape(1, T, 2)  # Shape: (1, T, 2)
            all_actions = np.concatenate((base_actions, noisy_actions), axis=0)  # Shape: (num_rollouts + 1, T, 2)
        else:
            all_actions = noisy_actions
        # Generate rollouts using the existing propagate function
        # clip the all_actions
        all_actions[:, :, 0] = np.clip(all_actions[:, :, 0], self.dtype(v_bounds[0]), self.dtype(v_bounds[1]))
        all_actions[:, :, 1] = np.clip(all_actions[:, :, 1], self.dtype(delta_bounds[0]), self.dtype(delta_bounds[1]))
        
        rollout_states = self.propagate_kernel(L, dt, x0, all_actions)
        
        return all_actions, rollout_states 

    def shift_and_update(self, new_x0, u_cur, num_shifts=1):
        self.curr_x = new_x0.copy()
        self.shift_optimal_control_sequence(u_cur, num_shifts)

    def shift_optimal_control_sequence(self, u_cur, num_shifts=1):
        u_cur_shifted = u_cur.copy()
        u_cur_shifted[:-num_shifts] = u_cur_shifted[num_shifts:]
        self.curr_u = u_cur_shifted.copy()




    # calcualte the cost of the final actions
