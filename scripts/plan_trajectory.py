#!/usr/bin/env python3
"""
Compute an optimal car trajectory from point A to point B under Ackermann
steering dynamics, constrained to stay on the drivable road area produced by
``extract_road_layout.py``.

Formulation (multiple shooting, CasADi / IPOPT)
------------------------------------------------
State    s = [x, z, θ, v]
  x, z  : 2-D position in KITTI world frame (+X right, +Z forward)
  θ     : yaw angle (rad) measured from +Z axis toward +X, CW positive
  v     : forward speed (m/s), constrained ≥ 0 (forward-only driving)

Control  u = [δ, a]
  δ     : front wheel steering angle (rad), |δ| ≤ δ_max
  a     : longitudinal acceleration (m/s²), a_min ≤ a ≤ a_max

Dynamics  Ackermann / planar bicycle model, integrated with RK4:
  ẋ = v · sin(θ)
  ż = v · cos(θ)
  θ̇ = v · tan(δ) / L       (L = wheelbase)
  v̇ = a

Road constraint
  The occupancy grid (``{seq}_occupancy_grid.npz``) from
  ``extract_road_layout.py`` is loaded and turned into a smooth CasADi
  bspline interpolant (after a small Gaussian blur to regularise the binary
  grid).  At every multiple-shooting node the interpolant value must be ≥
  road_threshold (default 0.5), i.e. the vehicle centre must stay in the
  drivable area.

Objective
  Minimise total arc-length (path distance) + optional centerline-tracking term
    J = Σ_{k=0}^{N-1}  v_k · dt
        + ε   · Σ_k (δ_k² + a_k²)
        + w_cl · Σ_k e_lat,k²
  The first term is arc-length = ∫|v|dt; the second is a tiny control
  regulariser to keep the problem well-conditioned; the third penalises
  signed lateral deviation from the resampled centerline in the Frenet frame:
    e_lat,k = cos(ψ_k)·(x_k - x̄_k) − sin(ψ_k)·(z_k − z̄_k)
  where (x̄_k, z̄_k) and ψ_k are the position and yaw of the k-th
  resampled centerline node.  This term prevents the optimizer from
  cutting corners while leaving arc-length minimisation as the primary goal.
  Set --w_centerline 0 (default) to disable it.

Endpoints  A and B
  A  = centerline node closest in arc-length to frame ``--start_frame``
  B  = centerline node closest in arc-length to frame
       ``--start_frame + --horizon_frames``
  The centerline graph (``{seq}_centerline_graph.npz``) supplies node
  positions and headings; ``--n_frames`` must match the value used when
  running ``extract_road_layout.py`` so that frame indices can be mapped to
  arc-lengths correctly.

Usage
-----
python scripts/plan_trajectory.py \\
    --occupancy_grid  /path/to/00_occupancy_grid.npz  \\
    --centerline      /path/to/00_centerline_graph.npz \\
    --output_dir      /path/to/output/ \\
    --n_frames        150  \\
    --start_frame     0    \\
    --horizon_frames  100

Key optional flags
------------------
--n_steps       Number of multiple-shooting intervals (default 50)
--dt            Integration time-step per interval (seconds, default 0.1)
--wheelbase     Vehicle wheelbase in metres (default 2.7)
--v_max         Maximum speed m/s (default 20.0)
--delta_max     Maximum steering angle rad (default 0.52 ≈ 30°)
--a_max / --a_min  Acceleration bounds (default +3 / -5 m/s²)
--road_threshold  Minimum interpolant value to count as drivable (default 0.5)
--sigma_blur    Gaussian sigma for grid pre-smoothing (default 2.0 cells)
--visualize     Save a top-down PNG of grid + trajectory (requires matplotlib)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Ackermann bicycle dynamics
# ---------------------------------------------------------------------------

def make_ackermann_rhs(L: float):
    """Return a CasADi function  f(s, u) -> ds/dt  for the bicycle model.

    State  s = [x, z, theta, v]
    Control u = [delta, a]
    """
    import casadi as ca

    s = ca.MX.sym("s", 4)
    u = ca.MX.sym("u", 2)
    x, z, theta, v = s[0], s[1], s[2], s[3]
    delta, a = u[0], u[1]

    dx     = v * ca.sin(theta)
    dz     = v * ca.cos(theta)
    dtheta = v * ca.tan(delta) / L
    dv     = a

    rhs = ca.vertcat(dx, dz, dtheta, dv)
    return ca.Function("f", [s, u], [rhs], ["s", "u"], ["ds_dt"])


def make_rk4_step(rhs_fn, dt: float):
    """Return a CasADi function  F(s, u) -> s_next  via RK4."""
    import casadi as ca

    s = ca.MX.sym("s", 4)
    u = ca.MX.sym("u", 2)

    k1 = rhs_fn(s,            u)
    k2 = rhs_fn(s + dt/2 * k1, u)
    k3 = rhs_fn(s + dt/2 * k2, u)
    k4 = rhs_fn(s + dt    * k3, u)

    s_next = s + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
    return ca.Function("F", [s, u], [s_next], ["s", "u"], ["s_next"])


# ---------------------------------------------------------------------------
# Road interpolant from occupancy grid
# ---------------------------------------------------------------------------

def build_road_interpolant(
    grid: np.ndarray,
    origin_x: float,
    origin_z: float,
    resolution: float,
    sigma_blur: float = 2.0,
    method: str = "bspline",
):
    """
    Build a CasADi 2-D interpolant that returns 1 inside the drivable area
    and 0 outside.

    The binary grid is blurred with a Gaussian of ``sigma_blur`` cells before
    building the interpolant.  This turns the sharp 0/1 boundary into a
    smooth gradient that IPOPT can follow efficiently and avoids the bspline
    overshoot artefacts that would appear with a raw binary input.

    Parameters
    ----------
    grid        : (n_rows, n_cols) bool / float array; True = drivable.
                  n_rows indexes the Z axis, n_cols indexes the X axis.
    origin_x/z  : world coordinates of cell (col=0, row=0) [metres].
    resolution  : metres per cell.
    sigma_blur  : Gaussian blur radius in grid cells (0 = no blur).
    method      : 'bspline' (smooth, recommended) or 'linear'.

    Returns
    -------
    road_fn   : CasADi interpolant  road_fn([x, z]) → float in [0, 1]
    x_grid    : (n_cols,) world X breakpoints
    z_grid    : (n_rows,) world Z breakpoints
    """
    import casadi as ca

    n_rows, n_cols = grid.shape
    x_grid = origin_x + np.arange(n_cols) * resolution   # (n_cols,)
    z_grid = origin_z + np.arange(n_rows) * resolution   # (n_rows,)

    g = grid.astype(np.float64)

    if sigma_blur > 0:
        try:
            from scipy.ndimage import gaussian_filter
            g = gaussian_filter(g, sigma=sigma_blur)
        except ImportError:
            print("[warn] scipy not available – using raw binary grid for "
                  "interpolant (may cause bspline overshoot).")

    # CasADi interpolant memory layout for grid=[x_breaks, z_breaks]:
    #   values[i * n_rows + j]  =  f(x_grid[i], z_grid[j])
    # where i indexes x (outer/slow) and j indexes z (inner/fast).
    # Our grid is shaped (n_rows, n_cols) = (n_z, n_x), so
    #   g[j, i] = f(x_grid[i], z_grid[j])
    # ⟹ values = g.T.ravel()  (T gives shape (n_cols, n_rows), then C-order
    #   flatten gives values[i*n_rows + j] = g.T[i, j] = g[j, i]  ✓)
    values = g.T.ravel()

    road_fn = ca.interpolant(
        "road",
        method,
        [x_grid.tolist(), z_grid.tolist()],
        values.tolist(),
    )
    return road_fn, x_grid, z_grid


def verify_road_interpolant(
    road_fn,
    x_test: float,
    z_test: float,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    grid: np.ndarray,
    sigma_blur: float,
    method: str,
) -> object:
    """
    Evaluate road_fn at a known drivable point.  If the value is < 0.1
    (suggesting the x/z layout is transposed), rebuild with the alternative
    memory layout and return the corrected interpolant.
    """
    import casadi as ca

    val = float(road_fn(ca.vertcat(x_test, z_test)))
    if val >= 0.1:
        return road_fn  # layout is correct

    print(f"[warn] Road interpolant returned {val:.4f} at the start node "
          f"(expected ≈ 1).  Retrying with transposed grid layout …")

    # Alternative layout: values = g.ravel() (x slow, z fast → transposed)
    g = grid.astype(np.float64)
    if sigma_blur > 0:
        try:
            from scipy.ndimage import gaussian_filter
            g = gaussian_filter(g, sigma=sigma_blur)
        except ImportError:
            pass
    values_alt = g.ravel()

    road_fn_alt = ca.interpolant(
        "road",
        method,
        [x_grid.tolist(), z_grid.tolist()],
        values_alt.tolist(),
    )
    val2 = float(road_fn_alt(ca.vertcat(x_test, z_test)))
    if val2 >= 0.1:
        print(f"[i] Transposed layout correct (value = {val2:.4f}).")
        return road_fn_alt

    print(f"[warn] Both grid layouts return non-drivable value at start point "
          f"({val2:.4f}).  The start point may not be exactly on the road; "
          f"proceeding with the original layout – check your inputs.")
    return road_fn


# ---------------------------------------------------------------------------
# Endpoint helpers
# ---------------------------------------------------------------------------

def frame_to_arc(frame_idx: int, n_frames: int, total_arc: float) -> float:
    """
    Map a frame index to an approximate arc-length position along the
    trajectory, assuming uniform frame spacing.

    Parameters
    ----------
    frame_idx   : 0-based frame index.
    n_frames    : total number of frames that were used to build the
                  centerline (= ``--n_frames`` passed to
                  ``extract_road_layout.py``).
    total_arc   : total arc-length of the centerline graph (metres).
    """
    if n_frames <= 1:
        return 0.0
    return float(frame_idx) / (n_frames - 1) * total_arc


def find_nearest_node(arc_lengths: np.ndarray, target_arc: float) -> int:
    """Return the index of the centerline node closest to ``target_arc``."""
    return int(np.argmin(np.abs(arc_lengths - target_arc)))


def resample_centerline_init(
    nodes: np.ndarray,
    arc_lengths: np.ndarray,
    start_node: int,
    goal_node: int,
    n_steps: int,
) -> np.ndarray:
    """
    Resample the centerline segment from ``start_node`` to ``goal_node``
    into exactly ``n_steps + 1`` evenly-spaced waypoints (by arc-length).

    Parameters
    ----------
    nodes        : (N_nodes, 3) centerline node world XYZ positions.
    arc_lengths  : (N_nodes,)  cumulative arc-length at each node.
    start_node   : index of the start node.
    goal_node    : index of the goal node.
    n_steps      : number of OCP intervals (returns n_steps + 1 points).

    Returns
    -------
    waypoints : (n_steps + 1, 2) array of (x, z) positions along the
                centerline, uniformly spaced in arc-length.
    """
    # Slice the centerline segment (support forward or backward indexing)
    if start_node <= goal_node:
        seg_nodes = nodes[start_node : goal_node + 1]          # (M, 3)
        seg_arc   = arc_lengths[start_node : goal_node + 1]    # (M,)
    else:
        seg_nodes = nodes[goal_node : start_node + 1][::-1]    # reversed
        seg_arc   = arc_lengths[goal_node : start_node + 1]    # (M,)
        seg_arc   = seg_arc[-1] - seg_arc[::-1]                # monotone ↑

    # Normalise arc-length to [0, 1]
    arc_norm = (seg_arc - seg_arc[0]) / max(seg_arc[-1] - seg_arc[0], 1e-9)

    # Uniformly sampled parameter values
    t = np.linspace(0.0, 1.0, n_steps + 1)

    x_init = np.interp(t, arc_norm, seg_nodes[:, 0])
    z_init = np.interp(t, arc_norm, seg_nodes[:, 2])

    return np.stack([x_init, z_init], axis=1)   # (n_steps + 1, 2)


# ---------------------------------------------------------------------------
# NLP setup and solve
# ---------------------------------------------------------------------------

def solve_trajectory(
    # road
    road_fn,
    # endpoints
    s_start: np.ndarray,   # (4,) [x, z, theta, v]
    x_goal: float,
    z_goal: float,
    # dynamics
    rk4_step,
    # horizon
    n_steps: int,
    dt: float,
    # bounds
    v_min: float,
    v_max: float,
    delta_max: float,
    a_min: float,
    a_max: float,
    # constraint
    road_threshold: float,
    # vehicle footprint
    half_width: float = 0.0,   # metres; 0 = centre-only (legacy)
    half_length: float = 0.0,  # metres; 0 = centre-only (legacy)
    # objective
    reg_u: float = 1e-4,
    # centerline-tracking penalty (Frenet lateral offset)
    cl_waypoints: np.ndarray | None = None,  # (n_steps+1, 2) resampled centerline (x, z)
    cl_yaw: np.ndarray | None = None,        # (n_steps+1,)  centerline yaw angles (rad)
    w_centerline: float = 0.0,
    # solver options
    ipopt_max_iter: int = 2000,
    ipopt_print_level: int = 0,
    # warm-start
    init_waypoints: np.ndarray | None = None,  # (n_steps+1, 2) (x, z) or None
) -> dict:
    """
    Build and solve the multiple-shooting NLP with CasADi's Opti interface.

    Returns a dict with keys:
        x_traj      (n_steps+1,)  world X coordinates
        z_traj      (n_steps+1,)  world Z coordinates
        theta_traj  (n_steps+1,)  yaw angles (rad)
        v_traj      (n_steps+1,)  forward speeds (m/s)
        delta_traj  (n_steps,)    steering angles (rad)
        a_traj      (n_steps,)    accelerations (m/s²)
        arc_length  float         total arc-length of solution (m)
        cost        float         NLP objective value
        status      str           solver return status
    """
    import casadi as ca

    opti = ca.Opti()

    # ---- Decision variables -----------------------------------------------
    S = opti.variable(4, n_steps + 1)   # states  (4 × N+1)
    U = opti.variable(2, n_steps)       # controls (2 × N)

    # Convenience column accessors
    x     = S[0, :]
    z     = S[1, :]
    theta = S[2, :]
    v     = S[3, :]
    delta = U[0, :]
    a     = U[1, :]

    # ---- Objective --------------------------------------------------------
    #   arc-length = Σ v_k · dt  (forward-only, so v ≥ 0 ⟹ |v| = v)
    #   + small regularisation on controls
    #   + optional Frenet lateral deviation from centerline
    #
    #   Frenet signed lateral offset at node k (right = positive):
    #     e_lat,k = cos(ψ_k)·(x_k − x̄_k) − sin(ψ_k)·(z_k − z̄_k)
    #   where ψ_k, x̄_k, z̄_k come from the resampled centerline passed in.
    #   In the KITTI XZ frame (+X right, +Z forward, yaw from +Z toward +X):
    #     fwd  = (sin ψ, cos ψ)  in (X, Z)
    #     right = (cos ψ, −sin ψ) in (X, Z)
    #   so the lateral component of (x − x̄, z − z̄) is:
    #     e_lat = cos(ψ)·(x−x̄) − sin(ψ)·(z−z̄)
    use_cl_penalty = (
        w_centerline > 0.0
        and cl_waypoints is not None
        and cl_yaw is not None
        and len(cl_waypoints) == n_steps + 1
        and len(cl_yaw) == n_steps + 1
    )
    cost = 0
    for k in range(n_steps):
        cost += v[k] * dt
        cost += reg_u * (delta[k]**2 + a[k]**2)
        if use_cl_penalty:
            cl_x_k   = float(cl_waypoints[k, 0])
            cl_z_k   = float(cl_waypoints[k, 1])
            cl_yaw_k = float(cl_yaw[k])
            e_lat = (ca.cos(cl_yaw_k) * (x[k] - cl_x_k)
                     - ca.sin(cl_yaw_k) * (z[k] - cl_z_k))
            cost += w_centerline * e_lat**2
    # Lateral penalty at the terminal node too (k = n_steps)
    if use_cl_penalty:
        cl_x_k   = float(cl_waypoints[n_steps, 0])
        cl_z_k   = float(cl_waypoints[n_steps, 1])
        cl_yaw_k = float(cl_yaw[n_steps])
        e_lat = (ca.cos(cl_yaw_k) * (x[n_steps] - cl_x_k)
                 - ca.sin(cl_yaw_k) * (z[n_steps] - cl_z_k))
        cost += w_centerline * e_lat**2
    opti.minimize(cost)

    # ---- Dynamics constraints (defect = 0) --------------------------------
    for k in range(n_steps):
        s_k      = S[:, k]
        u_k      = U[:, k]
        s_k_next = rk4_step(s_k, u_k)
        opti.subject_to(S[:, k + 1] == s_k_next)

    # ---- Road constraints (vehicle footprint) ---------------------------
    # Four corners of the vehicle rectangle in the KITTI XZ plane.
    # Coordinate frame: +X right, +Z forward; heading θ from +Z toward +X.
    #   forward unit  = (sin θ,  cos θ)  in (X, Z)
    #   right   unit  = (cos θ, -sin θ)  in (X, Z)
    # Corner offsets (body frame → world frame):
    #   corner = centre ± half_length·fwd ± half_width·right
    _hw = half_width
    _hl = half_length
    _footprint_offsets = [
        ( _hl,  _hw),   # front-right
        ( _hl, -_hw),   # front-left
        (-_hl,  _hw),   # rear-right
        (-_hl, -_hw),   # rear-left
    ] if (_hw > 0 or _hl > 0) else [(0.0, 0.0)]

    for k in range(n_steps + 1):
        th = theta[k]
        sin_th = ca.sin(th)
        cos_th = ca.cos(th)
        for (dl, dw) in _footprint_offsets:
            # world displacement of this corner relative to centre
            dx_corner = dl * sin_th + dw * cos_th   # X component
            dz_corner = dl * cos_th - dw * sin_th   # Z component
            opti.subject_to(
                road_fn(ca.vertcat(x[k] + dx_corner, z[k] + dz_corner))
                >= road_threshold
            )

    # ---- State / control bounds -------------------------------------------
    opti.subject_to(opti.bounded(v_min,    v,      v_max))
    opti.subject_to(opti.bounded(-delta_max, delta, delta_max))
    opti.subject_to(opti.bounded(a_min,    a,      a_max))

    # ---- Boundary conditions ----------------------------------------------
    # Initial state (fixed)
    opti.subject_to(S[:, 0] == s_start)

    # Terminal position (fixed x, z; free θ and v at goal)
    opti.subject_to(x[n_steps] == x_goal)
    opti.subject_to(z[n_steps] == z_goal)

    # ---- Warm-start initialisation ----------------------------------------
    # Prefer a centerline-following path (init_waypoints) when supplied;
    # fall back to a straight line from A to B otherwise.  Both use a
    # constant forward speed = path length / total time.
    if init_waypoints is not None and len(init_waypoints) == n_steps + 1:
        x_init = init_waypoints[:, 0].astype(float)
        z_init = init_waypoints[:, 1].astype(float)
    else:
        # Straight-line fallback
        alphas = np.linspace(0.0, 1.0, n_steps + 1)
        x_init = float(s_start[0]) + alphas * (x_goal - s_start[0])
        z_init = float(s_start[1]) + alphas * (z_goal - s_start[1])

    # Per-segment arc-length and constant speed along the initial path
    seg_lengths = np.hypot(np.diff(x_init), np.diff(z_init))   # (n_steps,)
    path_length = float(seg_lengths.sum()) if len(seg_lengths) else 1.0
    v_guess     = path_length / (n_steps * dt)
    v_guess     = float(np.clip(v_guess, v_min + 1e-3, v_max))

    # Per-node heading derived from the initial path geometry
    dx_seg = np.diff(x_init)   # (n_steps,)
    dz_seg = np.diff(z_init)   # (n_steps,)
    theta_seg = np.arctan2(dx_seg, dz_seg)               # (n_steps,)
    # Extend to n_steps+1: duplicate last heading for the terminal node
    theta_init = np.append(theta_seg, theta_seg[-1])

    for k in range(n_steps + 1):
        opti.set_initial(S[0, k], x_init[k])
        opti.set_initial(S[1, k], z_init[k])
        opti.set_initial(S[2, k], theta_init[k])
        opti.set_initial(S[3, k], v_guess)
    opti.set_initial(delta, 0.0)
    opti.set_initial(a,     0.0)

    # ---- Solver configuration ---------------------------------------------
    opts = {
        "ipopt.max_iter":    ipopt_max_iter,
        "ipopt.print_level": ipopt_print_level,
        "ipopt.tol":         1e-6,
        "ipopt.acceptable_tol": 1e-4,
        "print_time":        0 if ipopt_print_level == 0 else 1,
    }
    opti.solver("ipopt", opts)

    # ---- Solve ------------------------------------------------------------
    try:
        sol = opti.solve()
        status = "optimal"
    except RuntimeError as exc:
        # Retrieve the last iterate even on non-convergence so callers can
        # inspect / visualise what the solver found.
        print(f"[warn] IPOPT did not converge cleanly: {exc}")
        sol    = opti.debug
        status = "infeasible_or_max_iter"

    # ---- Extract solution -------------------------------------------------
    x_traj     = np.array(sol.value(x)).ravel()
    z_traj     = np.array(sol.value(z)).ravel()
    theta_traj = np.array(sol.value(theta)).ravel()
    v_traj     = np.array(sol.value(v)).ravel()
    delta_traj = np.array(sol.value(delta)).ravel()
    a_traj     = np.array(sol.value(a)).ravel()

    arc_length = float(np.sum(np.hypot(np.diff(x_traj), np.diff(z_traj))))
    cost_val   = float(sol.value(cost))

    return {
        "x_traj":     x_traj,
        "z_traj":     z_traj,
        "theta_traj": theta_traj,
        "v_traj":     v_traj,
        "delta_traj": delta_traj,
        "a_traj":     a_traj,
        "arc_length": arc_length,
        "cost":       cost_val,
        "status":     status,
        "n_steps":    n_steps,
        "dt":         dt,
    }


# ---------------------------------------------------------------------------
# Height snapping
# ---------------------------------------------------------------------------

def snap_y_to_road(
    x_traj: np.ndarray,
    z_traj: np.ndarray,
    nodes: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    """
    For each planned (x, z) waypoint find the k nearest centerline nodes
    (queried in the XZ plane only) and return the mean of their road-surface
    Y values as the ground-level Y coordinate.

    Parameters
    ----------
    x_traj, z_traj : (M,) planned trajectory coordinates.
    nodes          : (N, 3) centerline node positions (world XYZ) whose Y
                     has already been snapped to the road surface by
                     ``build_centerline_graph``.
    k              : number of neighbours to average (clamped to len(nodes)).

    Returns
    -------
    y_traj : (M,) road-surface Y coordinate per waypoint (KITTI +Y = down).
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("[warn] scipy not available – Y will be linearly interpolated "
              "between start and goal node heights.")
        y_start = nodes[0, 1]
        y_end   = nodes[-1, 1]
        return np.linspace(y_start, y_end, len(x_traj))

    xz_query = np.stack([x_traj, z_traj], axis=1)          # (M, 2)
    k_eff    = min(k, len(nodes))
    tree     = cKDTree(nodes[:, [0, 2]])                    # build on XZ only
    _, idx   = tree.query(xz_query, k=k_eff, workers=-1)
    if k_eff == 1:
        idx = idx[:, np.newaxis]
    y_traj = nodes[idx, 1].mean(axis=1)                     # (M,)
    return y_traj.astype(np.float64)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(out_dir: Path, result: dict, start_node: int, goal_node: int) -> None:
    """Save trajectory as both .npz and .json."""
    traj_xz = np.stack([result["x_traj"], result["z_traj"]], axis=1)  # (N+1, 2)

    npz_path = out_dir / "optimal_trajectory.npz"
    np.savez_compressed(
        str(npz_path),
        x=result["x_traj"],
        y=result["y_traj"],
        z=result["z_traj"],
        theta=result["theta_traj"],
        v=result["v_traj"],
        delta=result["delta_traj"],
        a=result["a_traj"],
        n_steps=np.array(result["n_steps"]),
        dt=np.array(result["dt"]),
        arc_length=np.array(result["arc_length"]),
    )
    print(f"    [✓] {npz_path}")

    json_path = out_dir / "optimal_trajectory.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "status":        result["status"],
                "arc_length_m":  result["arc_length"],
                "cost":          result["cost"],
                "n_steps":       result["n_steps"],
                "dt_s":          result["dt"],
                "total_time_s":  result["n_steps"] * result["dt"],
                "start_node":    start_node,
                "goal_node":     goal_node,
                "trajectory": {
                    "x":     result["x_traj"].tolist(),
                    "y":     result["y_traj"].tolist(),
                    "z":     result["z_traj"].tolist(),
                    "theta": result["theta_traj"].tolist(),
                    "v":     result["v_traj"].tolist(),
                    "delta": result["delta_traj"].tolist(),
                    "a":     result["a_traj"].tolist(),
                },
            },
            f,
            indent=2,
        )
    print(f"    [✓] {json_path}")


def visualize(
    out_dir: Path,
    grid: np.ndarray,
    origin_x: float,
    origin_z: float,
    resolution: float,
    nodes: np.ndarray,
    result: dict,
    start_node: int,
    goal_node: int,
    half_width: float = 0.0,
    half_length: float = 0.0,
) -> None:
    """Save a top-down PNG showing the occupancy grid, centerline, and optimal trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        print("[warn] matplotlib not available – skipping visualisation.")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    # Occupancy grid (flip rows so Z increases upward)
    n_rows, n_cols = grid.shape
    extent = [
        origin_x,
        origin_x + n_cols * resolution,
        origin_z,
        origin_z + n_rows * resolution,
    ]
    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="Greens",
        alpha=0.4,
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Centerline nodes
    cx = nodes[:, 0]
    cz = nodes[:, 2]
    ax.plot(cx, cz, "b--", linewidth=1.0, alpha=0.5, label="Centerline")

    # Start / goal markers
    ax.plot(nodes[start_node, 0], nodes[start_node, 2],
            "go", markersize=12, label=f"Start (node {start_node})", zorder=5)
    ax.plot(nodes[goal_node, 0], nodes[goal_node, 2],
            "rs", markersize=12, label=f"Goal  (node {goal_node})", zorder=5)

    # Optimal trajectory
    ax.plot(result["x_traj"], result["z_traj"],
            "r-", linewidth=1.5, label="Optimal trajectory", zorder=4)

    # Vehicle footprint rectangles every n_steps // 10 steps
    n_steps = result["n_steps"]
    footprint_stride = max(1, n_steps // 10)
    _hw = half_width
    _hl = half_length
    footprint_indices = list(range(0, n_steps + 1, footprint_stride))
    if footprint_indices[-1] != n_steps:
        footprint_indices.append(n_steps)

    if _hw > 0 or _hl > 0:
        cmap_fp = plt.cm.plasma
        n_fp = len(footprint_indices)
        for i, k in enumerate(footprint_indices):
            xc  = result["x_traj"][k]
            zc  = result["z_traj"][k]
            th  = result["theta_traj"][k]
            sin_th = np.sin(th)
            cos_th = np.cos(th)
            # Four corners: (dl, dw) combinations where dl = ±hl, dw = ±hw
            # world displacement: X = dl*sin + dw*cos, Z = dl*cos - dw*sin
            corners = np.array([
                [xc + _hl*sin_th + _hw*cos_th, zc + _hl*cos_th - _hw*sin_th],  # front-right
                [xc + _hl*sin_th - _hw*cos_th, zc + _hl*cos_th + _hw*sin_th],  # front-left
                [xc - _hl*sin_th - _hw*cos_th, zc - _hl*cos_th + _hw*sin_th],  # rear-left
                [xc - _hl*sin_th + _hw*cos_th, zc - _hl*cos_th - _hw*sin_th],  # rear-right
            ])
            color = cmap_fp(i / max(n_fp - 1, 1))
            poly = Polygon(corners, closed=True,
                           facecolor=color, edgecolor="black",
                           alpha=0.45, linewidth=0.8, zorder=3)
            ax.add_patch(poly)
        # Add a dummy patch for the legend
        legend_fp = mpatches.Patch(facecolor="mediumpurple", edgecolor="black",
                                   alpha=0.6, label="Vehicle footprint")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(legend_fp)
    else:
        # No footprint: fall back to heading arrows
        ax.quiver(
            result["x_traj"][::5],
            result["z_traj"][::5],
            np.sin(result["theta_traj"][::5]),
            np.cos(result["theta_traj"][::5]),
            color="red",
            scale=15,
            width=0.003,
            alpha=0.8,
        )
        handles, labels = ax.get_legend_handles_labels()

    ax.set_xlabel("X – world (m, right)")
    ax.set_ylabel("Z – world (m, forward)")
    ax.set_title(
        f"Optimal trajectory  |  arc-length = {result['arc_length']:.1f} m  "
        f"|  status = {result['status']}"
    )
    ax.legend(handles=handles, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    png_path = out_dir / "optimal_trajectory.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [✓] {png_path}")


def plot_control_profiles(out_dir: Path, result: dict) -> None:
    """Save a figure with three sub-plots: velocity, acceleration, steering angle."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available – skipping control profile plots.")
        return

    n_steps = result["n_steps"]
    dt      = result["dt"]

    # Time axes: states are at nodes 0..N, controls at intervals 0..N-1
    t_state   = np.arange(n_steps + 1) * dt          # (N+1,)
    t_control = (np.arange(n_steps) + 0.5) * dt      # (N,)  interval midpoints

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(
        f"Control profiles  |  arc-length = {result['arc_length']:.1f} m  "
        f"|  status = {result['status']}",
        fontsize=11,
    )

    # -- Velocity -----------------------------------------------------------
    ax = axes[0]
    ax.plot(t_state, result["v_traj"], color="steelblue", linewidth=1.8)
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocity")

    # -- Acceleration -------------------------------------------------------
    ax = axes[1]
    ax.plot(t_control, result["a_traj"], color="darkorange", linewidth=1.8)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Longitudinal Acceleration")

    # -- Steering angle -----------------------------------------------------
    ax = axes[2]
    ax.plot(t_control, np.degrees(result["delta_traj"]),
            color="forestgreen", linewidth=1.8)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Steering angle (deg)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Front-Wheel Steering Angle")

    fig.tight_layout()
    png_path = out_dir / "control_profiles.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [✓] {png_path}")


def save_trajectory_gif(
    out_dir: Path,
    grid: np.ndarray,
    origin_x: float,
    origin_z: float,
    resolution: float,
    nodes: np.ndarray,
    result: dict,
    start_node: int,
    goal_node: int,
    half_width: float = 0.0,
    half_length: float = 0.0,
    fps: int = 12,
) -> None:
    """Save a GIF animation with the ego vehicle moving along the trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.patches import Polygon
    except ImportError:
        print("[warn] matplotlib animation dependencies not available – skipping GIF export.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    # Occupancy grid (flip rows so Z increases upward)
    n_rows, n_cols = grid.shape
    extent = [
        origin_x,
        origin_x + n_cols * resolution,
        origin_z,
        origin_z + n_rows * resolution,
    ]
    ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap="Greens",
        alpha=0.35,
        vmin=0,
        vmax=1,
        aspect="equal",
    )

    # Centerline and full planned trajectory
    cx = nodes[:, 0]
    cz = nodes[:, 2]
    ax.plot(cx, cz, "b--", linewidth=1.0, alpha=0.5, label="Centerline")
    ax.plot(result["x_traj"], result["z_traj"],
            "r-", linewidth=1.3, alpha=0.8, label="Optimal trajectory")

    # Start / goal markers
    ax.plot(nodes[start_node, 0], nodes[start_node, 2],
            "go", markersize=10, label=f"Start (node {start_node})", zorder=5)
    ax.plot(nodes[goal_node, 0], nodes[goal_node, 2],
            "rs", markersize=10, label=f"Goal (node {goal_node})", zorder=5)

    # Animated artists: travelled path + ego marker/footprint
    (trail_line,) = ax.plot([], [], color="crimson", linewidth=2.3, zorder=6)
    (ego_dot,) = ax.plot([], [], "o", color="gold", markeredgecolor="black",
                         markersize=9, zorder=8)

    ego_poly = None
    if half_width > 0 or half_length > 0:
        ego_poly = Polygon(np.zeros((4, 2)), closed=True,
                           facecolor="gold", edgecolor="black",
                           alpha=0.65, linewidth=1.0, zorder=7)
        ax.add_patch(ego_poly)

    ax.set_xlabel("X – world (m, right)")
    ax.set_ylabel("Z – world (m, forward)")
    ax.set_title("Trajectory animation")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    # Keep fixed limits to avoid camera jitter during animation
    x_all = np.concatenate([result["x_traj"], cx])
    z_all = np.concatenate([result["z_traj"], cz])
    x_pad = max(2.0, 0.03 * (x_all.max() - x_all.min() + 1e-9))
    z_pad = max(2.0, 0.03 * (z_all.max() - z_all.min() + 1e-9))
    ax.set_xlim(float(x_all.min() - x_pad), float(x_all.max() + x_pad))
    ax.set_ylim(float(z_all.min() - z_pad), float(z_all.max() + z_pad))

    x_traj = result["x_traj"]
    z_traj = result["z_traj"]
    th_traj = result["theta_traj"]
    n_frames = len(x_traj)

    def _update(frame_idx: int):
        trail_line.set_data(x_traj[: frame_idx + 1], z_traj[: frame_idx + 1])
        xc = float(x_traj[frame_idx])
        zc = float(z_traj[frame_idx])
        th = float(th_traj[frame_idx])
        ego_dot.set_data([xc], [zc])

        if ego_poly is not None:
            sin_th = np.sin(th)
            cos_th = np.cos(th)
            corners = np.array([
                [xc + half_length * sin_th + half_width * cos_th, zc + half_length * cos_th - half_width * sin_th],
                [xc + half_length * sin_th - half_width * cos_th, zc + half_length * cos_th + half_width * sin_th],
                [xc - half_length * sin_th - half_width * cos_th, zc - half_length * cos_th + half_width * sin_th],
                [xc - half_length * sin_th + half_width * cos_th, zc - half_length * cos_th - half_width * sin_th],
            ])
            ego_poly.set_xy(corners)
            return trail_line, ego_dot, ego_poly

        return trail_line, ego_dot

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=max(1, int(1000 / max(fps, 1))),
        blit=True,
        repeat=True,
    )

    gif_path = out_dir / "optimal_trajectory.gif"
    try:
        writer = animation.PillowWriter(fps=max(fps, 1))
        anim.save(str(gif_path), writer=writer)
        print(f"    [✓] {gif_path}")
    except Exception as exc:
        print(f"[warn] Failed to save GIF animation: {exc}")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute an optimal car trajectory from A to B on the drivable "
            "road using Ackermann dynamics and multiple-shooting OCP (CasADi/IPOPT)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Inputs ---
    p.add_argument(
        "--occupancy_grid", required=True,
        help="Path to the occupancy grid .npz produced by extract_road_layout.py "
             "(e.g. 00_occupancy_grid.npz).",
    )
    p.add_argument(
        "--centerline", required=True,
        help="Path to the centerline graph .npz produced by extract_road_layout.py "
             "(e.g. 00_centerline_graph.npz).",
    )

    # --- Output ---
    p.add_argument(
        "--output_dir", required=True,
        help="Directory to write optimal_trajectory.npz / .json / .png.",
    )

    # --- Endpoint selection ---
    p.add_argument(
        "--n_frames", type=int, default=150,
        help="Number of frames used when running extract_road_layout.py "
             "(needed to map frame indices to arc-lengths).",
    )
    p.add_argument(
        "--start_frame", type=int, default=0,
        help="Frame index of the start point A.",
    )
    p.add_argument(
        "--horizon_frames", type=int, default=100,
        help="Number of frames between A and B.  Goal B = frame start_frame + horizon_frames.",
    )

    # --- OCP horizon ---
    p.add_argument(
        "--n_steps", type=int, default=50,
        help="Number of multiple-shooting intervals.",
    )
    p.add_argument(
        "--dt", type=float, default=0.1,
        help="Integration time-step per interval (seconds).",
    )

    # --- Vehicle ---
    p.add_argument(
        "--wheelbase", type=float, default=2.7,
        help="Vehicle wheelbase L (metres).",
    )
    p.add_argument(
        "--v_max", type=float, default=20.0,
        help="Maximum forward speed (m/s).",
    )
    p.add_argument(
        "--v_min", type=float, default=0.0,
        help="Minimum forward speed (m/s); set to 0 to allow stopping.",
    )
    p.add_argument(
        "--delta_max", type=float, default=0.524,
        help="Maximum absolute steering angle (rad).  Default ≈ 30°.",
    )
    p.add_argument(
        "--a_max", type=float, default=3.0,
        help="Maximum acceleration (m/s²).",
    )
    p.add_argument(
        "--a_min", type=float, default=-5.0,
        help="Maximum deceleration (m/s², should be negative).",
    )

    # --- Vehicle footprint ---
    p.add_argument(
        "--half_width", type=float, default=1.0,
        help="Vehicle half-width (metres).  Road constraints are checked at all "
             "four corners of the vehicle rectangle.  Set to 0 to revert to "
             "centre-only behaviour.",
    )
    p.add_argument(
        "--half_length", type=float, default=2.5,
        help="Vehicle half-length (metres, front/rear from centre).  "
             "Combined with --half_width to define the four footprint corners.",
    )

    # --- Road constraint ---
    p.add_argument(
        "--road_threshold", type=float, default=0.5,
        help="Minimum interpolant value for a position to be considered drivable.",
    )
    p.add_argument(
        "--sigma_blur", type=float, default=2.0,
        help="Gaussian blur sigma applied to the binary grid before building "
             "the CasADi bspline interpolant (grid cells).  Larger values make "
             "the boundary softer and easier for IPOPT to handle; 0 disables "
             "blurring.",
    )

    # --- Objective ---
    p.add_argument(
        "--reg_u", type=float, default=1e-4,
        help="Control regularisation weight (applied to δ² + a² at each step).",
    )
    p.add_argument(
        "--w_centerline", type=float, default=0.1,
        help="Weight for the Frenet-frame lateral offset penalty from the "
             "centerline (applied to e_lat² at each node).  Set to 0 to disable. "
             "Typical range: 0.05–1.0.  Higher values enforce tighter lane-keeping "
             "at the cost of slightly increased arc-length.",
    )

    # --- Solver ---
    p.add_argument(
        "--max_iter", type=int, default=2000,
        help="IPOPT maximum iterations.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print IPOPT iteration log (print_level = 5).",
    )

    # --- Optional output ---
    p.add_argument(
        "--visualize", action="store_true",
        help="Save a top-down PNG visualisation of the trajectory.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Check CasADi availability ----------------------------------------
    try:
        import casadi  # noqa: F401
    except ImportError:
        sys.exit(
            "[error] CasADi is not installed.  Install with:\n"
            "    pip install casadi"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load occupancy grid ----------------------------------------------
    occ_path = Path(args.occupancy_grid)
    if not occ_path.exists():
        sys.exit(f"[error] Occupancy grid not found: {occ_path}")

    occ = np.load(str(occ_path))
    grid       = occ["grid"].astype(bool)           # (n_rows, n_cols)
    origin_x   = float(occ["origin_x"])
    origin_z   = float(occ["origin_z"])
    resolution = float(occ["resolution"])
    n_rows, n_cols = grid.shape

    print(f"[i] Occupancy grid: {n_rows} × {n_cols} cells, "
          f"resolution = {resolution} m, "
          f"origin = ({origin_x:.2f}, {origin_z:.2f}) m, "
          f"drivable cells = {int(grid.sum()):,}")

    # ---- Load centerline graph --------------------------------------------
    cl_path = Path(args.centerline)
    if not cl_path.exists():
        sys.exit(f"[error] Centerline graph not found: {cl_path}")

    cl          = np.load(str(cl_path))
    nodes       = cl["nodes"]       # (N_nodes, 3)  world XYZ
    yaw         = cl["yaw"]         # (N_nodes,)    heading from +Z
    arc_lengths = cl["arc_length"]  # (N_nodes,)    cumulative arc-length

    n_nodes    = len(nodes)
    total_arc  = float(arc_lengths[-1])
    print(f"[i] Centerline: {n_nodes} nodes, total arc-length = {total_arc:.1f} m")

    # ---- Determine start and goal nodes -----------------------------------
    start_arc = frame_to_arc(args.start_frame, args.n_frames, total_arc)
    goal_arc  = frame_to_arc(
        args.start_frame + args.horizon_frames, args.n_frames, total_arc
    )
    # Clamp goal arc to the available trajectory length
    goal_arc  = float(np.clip(goal_arc, 0.0, total_arc))

    start_node = find_nearest_node(arc_lengths, start_arc)
    goal_node  = find_nearest_node(arc_lengths, goal_arc)

    if start_node == goal_node:
        sys.exit(
            f"[error] Start and goal map to the same centerline node ({start_node}). "
            f"Increase --horizon_frames or check --n_frames."
        )

    s_start = np.array([
        nodes[start_node, 0],   # x
        nodes[start_node, 2],   # z  (KITTI: Y is up → index 2 is Z)
        float(yaw[start_node]), # theta
        0.0,                    # v  (start from rest)
    ], dtype=float)
    x_goal = float(nodes[goal_node, 0])
    z_goal = float(nodes[goal_node, 2])

    print(f"[i] Start node {start_node}: "
          f"A = ({s_start[0]:.2f}, {s_start[1]:.2f}) m, "
          f"θ = {np.degrees(s_start[2]):.1f}°")
    print(f"[i] Goal  node {goal_node}: "
          f"B = ({x_goal:.2f}, {z_goal:.2f}) m")
    straight_dist = float(np.hypot(x_goal - s_start[0], z_goal - s_start[1]))
    print(f"[i] Straight-line A→B = {straight_dist:.1f} m, "
          f"OCP horizon = {args.n_steps * args.dt:.1f} s")

    # Warn if the straight-line distance is physically unreachable with v_max
    max_reachable = args.v_max * args.n_steps * args.dt
    if straight_dist > max_reachable:
        print(
            f"[warn] Straight-line distance ({straight_dist:.1f} m) exceeds "
            f"v_max × T = {max_reachable:.1f} m.  "
            f"Increase --v_max, --n_steps, or --dt."
        )

    # ---- Build road interpolant ------------------------------------------
    print("[i] Building road interpolant …")
    road_fn, x_grid_arr, z_grid_arr = build_road_interpolant(
        grid, origin_x, origin_z, resolution,
        sigma_blur=args.sigma_blur,
        method="bspline",
    )
    # Sanity check at start node
    road_fn = verify_road_interpolant(
        road_fn,
        s_start[0], s_start[1],
        x_grid_arr, z_grid_arr,
        grid, args.sigma_blur, "bspline",
    )

    # ---- Build dynamics functions ----------------------------------------
    rhs_fn   = make_ackermann_rhs(args.wheelbase)
    rk4_step = make_rk4_step(rhs_fn, args.dt)

    # ---- Resample centerline as warm-start waypoints ---------------------
    print("[i] Building centerline warm-start …")
    init_waypoints = resample_centerline_init(
        nodes, arc_lengths, start_node, goal_node, args.n_steps
    )
    cl_arc = float(np.sum(np.hypot(np.diff(init_waypoints[:, 0]),
                                    np.diff(init_waypoints[:, 1]))))
    print(f"[i] Warm-start path arc-length = {cl_arc:.1f} m  "
          f"({args.n_steps + 1} waypoints along centerline)")

    # Derive yaw angles for each resampled centerline node from finite differences.
    # This gives the heading ψ_k at the k-th node, used for the Frenet lateral
    # offset penalty: e_lat,k = cos(ψ_k)·(x_k−x̄_k) − sin(ψ_k)·(z_k−z̄_k).
    dx_cl = np.diff(init_waypoints[:, 0])   # (n_steps,)
    dz_cl = np.diff(init_waypoints[:, 1])   # (n_steps,)
    cl_yaw_seg = np.arctan2(dx_cl, dz_cl)                     # (n_steps,)
    cl_yaw_resampled = np.append(cl_yaw_seg, cl_yaw_seg[-1])  # (n_steps+1,)

    # ---- Solve the OCP ---------------------------------------------------
    print(f"[i] Solving OCP (N = {args.n_steps} steps, dt = {args.dt} s, "
          f"total horizon = {args.n_steps * args.dt:.1f} s) …")

    print(f"[i] Vehicle footprint: half_width = {args.half_width} m, "
          f"half_length = {args.half_length} m  "
          f"({'4-corner check' if args.half_width > 0 or args.half_length > 0 else 'centre-only'})")

    cl_penalty_str = (f"w_centerline = {args.w_centerline}"
                      if args.w_centerline > 0 else "disabled")
    print(f"[i] Centerline lateral penalty: {cl_penalty_str}")

    result = solve_trajectory(
        road_fn        = road_fn,
        s_start        = s_start,
        x_goal         = x_goal,
        z_goal         = z_goal,
        rk4_step       = rk4_step,
        n_steps        = args.n_steps,
        dt             = args.dt,
        v_min          = args.v_min,
        v_max          = args.v_max,
        delta_max      = args.delta_max,
        a_min          = args.a_min,
        a_max          = args.a_max,
        road_threshold = args.road_threshold,
        half_width     = args.half_width,
        half_length    = args.half_length,
        reg_u          = args.reg_u,
        cl_waypoints   = init_waypoints,
        cl_yaw         = cl_yaw_resampled,
        w_centerline   = args.w_centerline,
        ipopt_max_iter    = args.max_iter,
        ipopt_print_level = 5 if args.verbose else 0,
        init_waypoints    = init_waypoints,
    )

    print(f"\n[i] Solver status  : {result['status']}")
    print(f"[i] Arc-length     : {result['arc_length']:.2f} m  "
          f"(straight-line = {straight_dist:.2f} m, "
          f"ratio = {result['arc_length'] / (straight_dist + 1e-9):.3f})")
    print(f"[i] Speed range    : {result['v_traj'].min():.2f} – "
          f"{result['v_traj'].max():.2f} m/s")
    print(f"[i] Steering range : {np.degrees(result['delta_traj'].min()):.1f}° – "
          f"{np.degrees(result['delta_traj'].max()):.1f}°")

    # ---- Snap Y to road surface ------------------------------------------
    print("[i] Snapping trajectory Y to road surface …")
    y_traj = snap_y_to_road(result["x_traj"], result["z_traj"], nodes)
    result["y_traj"] = y_traj
    print(f"[i] Y range        : {y_traj.min():.3f} – {y_traj.max():.3f} m  "
          f"(KITTI +Y = down)")

    # ---- Save results ----------------------------------------------------
    print("\n[i] Saving outputs …")
    save_results(out_dir, result, start_node, goal_node)

    if args.visualize:
        print("[i] Rendering visualisation …")
        visualize(
            out_dir, grid, origin_x, origin_z, resolution,
            nodes, result, start_node, goal_node,
            half_width=args.half_width,
            half_length=args.half_length,
        )
        print("[i] Rendering trajectory GIF …")
        save_trajectory_gif(
            out_dir, grid, origin_x, origin_z, resolution,
            nodes, result, start_node, goal_node,
            half_width=args.half_width,
            half_length=args.half_length,
        )
        print("[i] Rendering control profiles …")
        plot_control_profiles(out_dir, result)

    print(f"\n[✓] Done.  Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
