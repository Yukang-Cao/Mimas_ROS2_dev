import numpy as np
import yaml
import numpy as np
import os
'''
Single-track model of the mobile robot and kinematic model of the mobile robot 
'''
DTYPE = np.float32
'''
Kinematic model of the mobile robot
x = [x, y, theta]
u = [v, delta] # linear and steering angle
x_next = x + dt * [v * cos(theta), v * sin(theta), v * tan(delta) / L]
init values from f1tenth
'''
class Vehicle:
    def __init__(self, wheelbase=0.3, length=0.57, width=0.3, dt=0.05, v_bounds=(0.01, 2.5), delta_bounds=(-np.pi/6, np.pi/6)):
        self.wheelbase = DTYPE(wheelbase)
        self.length = DTYPE(length)
        self.width = DTYPE(width)
        self.dt = DTYPE(dt)
        self.v_bounds = v_bounds
        self.delta_bounds = delta_bounds
    
    @property
    def vehicle_type(self):
        return "kinematic"

class VehicleSingleTrack:
    """
    Dynamic single-track (bicycle) model with small-angle linear tire forces.
    State x = [x, y, v, psi, r, beta]
        x    : position [m]
        y    : position [m]
        v    : speed (longitudinal at CoM) [m/s]
        psi  : yaw [rad]
        r    : yaw rate psi_dot [rad/s]
        beta : body slip at CoM [rad]
    Input u = [a_long, delta_rate]
        a_long     : desired longitudinal acceleration [m/s^2]
        delta_rate : desired steering angle rate [rad/s]
    Steering state is δ (front steer); it’s kept internally and exposed as self.delta.
    Integration: forward Euler with dt.
    """
    def __init__(
        self,
        dt=0.05,
        # geometry
        lf=0.15, lr=0.15,           # [m] front/rear CG distances (lf+lr = wheelbase)
        length=0.57, width=0.30,    # [m] body box, for footprint if needed
        # mass properties
        m=3.5,                      # [kg]
        Iz=0.06,                    # [kg m^2] (small RC car-ish)
        h_cg=0.05,                  # [m] CG height (used only in friction circle helper below)
        # tire (linear cornering stiffness split into CS_f, CS_r with friction coeff mu)
        CSf=20.0, CSr=20.0,         # [1/rad] cornering stiffness coefficients (scaled by mu below)
        mu=1.0,                     # friction coefficient
        # limits
        v_bounds=(0.0, 5.0),        # [m/s]
        delta_bounds=(-np.pi/3, np.pi/3),     # [rad]
        delta_rate_bounds=(-6.0, 6.0),        # [rad/s]
        a_long_min=-8.0, a_long_max=8.0,      # [m/s^2]
        a_lat_max=None,             # [m/s^2] if None, use mu*g
        # numerics
        clip_friction_circle=True,
        dtype=DTYPE,
    ):
        self.dt = dtype(dt)
        self.lf, self.lr = dtype(lf), dtype(lr)
        self.length, self.width = dtype(length), dtype(width)
        self.m, self.Iz, self.h_cg = dtype(m), dtype(Iz), dtype(h_cg)
        self.CSf, self.CSr, self.mu = dtype(CSf), dtype(CSr), dtype(mu)

        self.v_bounds = (dtype(v_bounds[0]), dtype(v_bounds[1]))
        self.delta_bounds = (dtype(delta_bounds[0]), dtype(delta_bounds[1]))
        self.delta_rate_bounds = (dtype(delta_rate_bounds[0]), dtype(delta_rate_bounds[1]))
        self.a_long_min, self.a_long_max = dtype(a_long_min), dtype(a_long_max)

        self.a_lat_max = dtype(a_lat_max) if a_lat_max is not None else dtype(9.81) * self.mu
        self.clip_friction_circle = clip_friction_circle

        # steering state
        self.delta = dtype(0.0)

    @property
    def wheelbase(self):
        return self.lf + self.lr
    
    @property
    def vehicle_type(self):
        return "singletrack"

    def _clamp(self, val, lo, hi):
        return np.minimum(np.maximum(val, lo), hi)

    def reset(self, x0=np.zeros(6, dtype=DTYPE), delta0=0.0):
        x0 = np.asarray(x0, dtype=DTYPE)
        assert x0.shape == (6,)
        self.delta = self._clamp(DTYPE(delta0), *self.delta_bounds)
        return x0.copy()

    def step(self, x, u):
        """
        One Euler integration step.
        x: (6,) -> [x, y, v, psi, r, beta]
        u: (2,) -> [a_long_des, delta_rate_des]
        returns x_next
        """
        x = np.asarray(x, dtype=DTYPE)
        u = np.asarray(u, dtype=DTYPE)

        px, py, v, psi, r, beta = x
        a_long_des, delta_rate_des = u

        # 1) clamp inputs
        delta_rate = self._clamp(delta_rate_des, *self.delta_rate_bounds)
        a_long = self._clamp(a_long_des, self.a_long_min, self.a_long_max)

        # 2) integrate steering with rate & clamp angle
        self.delta = self._clamp(self.delta + self.dt * delta_rate, *self.delta_bounds)
        delta = self.delta

        # 3) lateral tire "gains" (linear tire forces with split CS & mu)
        #    We separate mu and CS as in CommonRoad ST formulation.
        #    Effective cornering stiffness terms (proportional to normal load) are merged here.
        #    For simplicity we assume nominal vertical loads -> use coefficients directly.
        #    (For higher fidelity, include load transfer; omitted here for speed/clarity.)
        mu = self.mu
        CSf = self.CSf
        CSr = self.CSr
        lf, lr = self.lf, self.lr
        m, Iz = self.m, self.Iz

        # 4) friction circle (optional): limit a_long so that sqrt(a_long^2 + (v*r)^2) <= a_max
        if self.clip_friction_circle:
            a_lat = v * r
            rad = np.sqrt(a_long * a_long + a_lat * a_lat)
            if rad > self.a_lat_max and rad > 1e-6:
                scale = self.a_lat_max / rad
                a_long *= scale  # keep same direction, shrink magnitude

        # 5) dynamics (single-track with β and r)
        #    βdot and rdot from linear bicycle (no load transfer, small angles)
        #    See CommonRoad ST (state includes v, ψ, r, β and steering via δ).  :contentReference[oaicite:1]{index=1}
        v_safe = np.maximum(DTYPE(0.1), np.abs(v))  # avoid singularity at very low speed

        # β̇
        beta_dot = (
            mu / (v_safe * (lr + lf))
            * (
                CSf * delta
                - (CSr + CSf) * beta
                + (CSr * lr - CSf * lf) * (r / v_safe)
            )
            - r
        )

        # ṙ
        r_dot = (
            mu * m / (Iz * (lr + lf))
            * (
                lf * CSf * delta
                + (CSr * lr - CSf * lf) * beta
                - (CSf * lf * lf + CSr * lr * lr) * (r / v_safe)
            )
        )

        # v̇ and kinematics
        v_dot = a_long
        psi_dot = r
        # world-frame position using (psi + beta)
        px_dot = v * np.cos(psi + beta)
        py_dot = v * np.sin(psi + beta)

        # 6) integrate
        px_next  = px  + self.dt * px_dot
        py_next  = py  + self.dt * py_dot
        v_next   = self._clamp(v + self.dt * v_dot, *self.v_bounds)
        psi_next = psi + self.dt * psi_dot
        r_next   = r   + self.dt * r_dot
        beta_next= beta+ self.dt * beta_dot

        return np.array([px_next, py_next, v_next, psi_next, r_next, beta_next], dtype=DTYPE)

    # Convenience: run multiple steps
    def rollout(self, x0, U):
        """
        U: (T,2) sequence of [a_long, delta_rate]
        returns X: (T+1,6)
        """
        x = np.asarray(x0, dtype=DTYPE)
        X = [x.copy()]
        for u in np.asarray(U, dtype=DTYPE):
            x = self.step(x, u)
            X.append(x)
        return np.stack(X, axis=0)

    # Optional: expose current steering
    def get_delta(self):
        return float(self.delta)




DTYPE = np.float32

def load_config_yaml(path_or_cfg):
    """
    Accepts either:
      - str / os.PathLike -> YAML file path
      - dict              -> already-parsed config (returns as-is)
      - file-like object  -> open file handle
    """
    if isinstance(path_or_cfg, dict):
        return path_or_cfg
    if hasattr(path_or_cfg, "read"):  # file-like
        return yaml.safe_load(path_or_cfg.read())
    if isinstance(path_or_cfg, (str, os.PathLike)):
        with open(path_or_cfg, "r") as f:
            return yaml.safe_load(f)
    raise TypeError(
        "load_config_yaml expected a path, dict, or file-like object, "
        f"got {type(path_or_cfg).__name__}"
    )


def vehicle_from_config(path, model_type="kinematic"):
    """
    Factory: create a Vehicle or VehicleSingleTrack from YAML config.

    Args:
        path (str): path to yaml file
        model_type (str): "kinematic" or "singletrack"
    """
    cfg = load_config_yaml(path)

    if model_type.lower() == "kinematic":
        return Vehicle(
            wheelbase=cfg.get("wheelbase", 0.3),
            length=cfg.get("length", 0.57),
            width=cfg.get("width", 0.3),
            dt=cfg.get("dt", 0.05),
            v_bounds=tuple(cfg.get("v_bounds", (0.01, 2.5))),
            delta_bounds=tuple(cfg.get("delta_bounds", (-np.pi/6, np.pi/6))),
        )

    elif model_type.lower() in ["singletrack", "bicycle"]:
        return VehicleSingleTrack(
            dt=cfg.get("dt", 0.05),
            lf=cfg.get("lf", 0.15),
            lr=cfg.get("lr", 0.15),
            length=cfg.get("length", 0.57),
            width=cfg.get("width", 0.3),
            m=cfg.get("m", 3.5),
            Iz=cfg.get("Iz", 0.06),
            h_cg=cfg.get("h_cg", 0.05),
            CSf=cfg.get("CSf", 20.0),
            CSr=cfg.get("CSr", 20.0),
            mu=cfg.get("mu", 1.0),
            v_bounds=tuple(cfg.get("v_bounds", (0.0, 5.0))),
            delta_bounds=tuple(cfg.get("delta_bounds", (-np.pi/3, np.pi/3))),
            delta_rate_bounds=tuple(cfg.get("delta_rate_bounds", (-6.0, 6.0))),
            a_long_min=cfg.get("a_long_min", -8.0),
            a_long_max=cfg.get("a_long_max", 8.0),
            a_lat_max=cfg.get("a_lat_max", None),
            clip_friction_circle=cfg.get("clip_friction_circle", True),
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")


if __name__ == "__main__":
    cfg = load_config_yaml("/mnt/c/Users/ogpoy/Documents/GitHub/cuniform/uncertainity_aware_exploraration/config/vehicle_parameters.yaml")
    vehicle = vehicle_from_config(cfg, model_type="singletrack")
    print(np.round(vehicle.wheelbase, 3))
