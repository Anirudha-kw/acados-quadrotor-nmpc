import pybullet as p
import pybullet_data
import numpy as np
import time
import math
import random

from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_model import quadrotor_model_auto
import matplotlib.pyplot as plt


# ---------- Turbulence and Wind ----------
# I referenced the following sources for designing the wind model: 1. https://en.wikipedia.org/wiki/Dryden_Wind_Turbulence_Model




class TurbulenceModel:
    def __init__(self,
                 sigma=np.array([0.2, 0.2, 0.1]),
                 L=np.array([25.0, 25.0, 15.0]),
                 airspeed_ref=2.0,
                 seed=42):
        self.sigma = np.array(sigma, dtype=float)
        self.L = np.array(L, dtype=float)
        self.airspeed_ref = float(airspeed_ref)
        self.state = np.zeros(3)
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.state[:] = 0.0

    def update(self, dt, t):
        """Update turbulence state and return velocity perturbation (m/s)"""
        V = max(self.airspeed_ref, 0.2)
        tau = np.maximum(self.L / V, 0.1)
        a = np.clip(1.0 - dt / tau, 0.0, 0.999)
        b = np.sqrt(np.maximum(2.0 * dt / tau, 1e-6))
        w = self.rng.randn(3)
        self.state = a * self.state + b * self.sigma * w
        return self.state.copy()


class WindDisturbance:
    """Physics-based wind model with aerodynamic drag"""
    def __init__(self,
                 mode="mixed",
                 const_wind_vel=np.array([1.8, 1.0, 2.0]),
                 sin_amp_vel=np.array([0.4, 1.2, .0]),
                 sin_freq=0.2,
                 gust_max_vel=np.array([1.5, 1.0, 0.8]),
                 gust_pulse_dur=0.5,
                 gust_cooldown=(2.0, 4.0),
                 air_density=1.225,
                 drag_cd=1.2,
                 drag_area=0.010,
                 turbulence: TurbulenceModel = None,
                 enable_momentum_disturbances=True,
                 momentum_disturbance_scale=0.01):
        self.mode = mode
        self.const_wind_vel = np.array(const_wind_vel, dtype=float)
        self.sin_amp_vel = np.array(sin_amp_vel, dtype=float)
        self.sin_freq = float(sin_freq)
        self.gust_max_vel = np.array(gust_max_vel, dtype=float)
        self.gust_pulse_dur = float(gust_pulse_dur)
        self.gust_cooldown = gust_cooldown
        self.air_density = float(air_density)
        self.drag_cd = float(drag_cd)
        self.drag_area = float(drag_area)
        self.enable_momentum_disturbances = enable_momentum_disturbances
        self.momentum_disturbance_scale = momentum_disturbance_scale
        
        if turbulence is None:
            turbulence = TurbulenceModel(
                sigma=np.array([0.2, 0.2, 0.1]),
                L=np.array([25.0, 25.0, 15.0]),
                airspeed_ref=2.0
            )
        self.turbulence = turbulence
        
        self._gust_active = False
        self._gust_until = 0.0
        self._gust_vel = np.zeros(3)
        self._next_gust_time = 0.5
        self._turb_vel = np.zeros(3)

    def reset(self, t0=0.0):
        self._gust_active = False
        self._gust_until = 0.0
        self._gust_vel[:] = 0.0
        self._next_gust_time = t0 + random.uniform(*self.gust_cooldown)
        self.turbulence.reset()
        self._turb_vel[:] = 0.0

    def _gust_update(self, t):
        """Update random gust state machine"""
        if self._gust_active:
            if t >= self._gust_until:
                self._gust_active = False
                self._next_gust_time = t + random.uniform(*self.gust_cooldown)
                self._gust_vel[:] = 0.0
        else:
            if t >= self._next_gust_time:
                self._gust_active = True
                self._gust_until = t + self.gust_pulse_dur
                self._gust_vel = (np.random.uniform(-1.0, 1.0, size=3) * self.gust_max_vel)

    def wind_velocity_field(self, pos_world, t):
        """Return total wind velocity at position (m/s) in WORLD frame"""
        v_wind = np.zeros(3)
        
        if self.mode in ("constant", "mixed"):
            v_wind += self.const_wind_vel
            
        if self.mode in ("sinusoid", "mixed"):
            v_wind += self.sin_amp_vel * math.sin(2.0 * math.pi * self.sin_freq * t)
            
        if self.mode in ("random_gusts", "mixed"):
            self._gust_update(t)
            v_wind += self._gust_vel
            
        if self.mode in ("turbulence", "mixed"):
            self._turb_vel = self.turbulence.update(0.02, t)
            v_wind += self._turb_vel
            
        return v_wind

    def get_drag_force(self, vel_world, wind_velocity_world):
        """Calculate aerodynamic drag force"""
        v_rel = vel_world - wind_velocity_world
        speed = np.linalg.norm(v_rel)
        
        if speed < 1e-6:
            return np.zeros(3)
        
        drag_magnitude = 0.5 * self.air_density * self.drag_cd * self.drag_area * speed
        F_drag = -drag_magnitude * v_rel
        
        return F_drag

    def get_momentum_disturbance(self):
        """Optional small direct force impacts"""
        if not self.enable_momentum_disturbances:
            return np.zeros(3)
        
        F_momentum = np.zeros(3)
        if self._gust_active:
            F_momentum = self.momentum_disturbance_scale * np.sign(self._gust_vel)
        
        return F_momentum

    def get_total_force(self, t, dt, pos_world, vel_world):
        """Return total external force from wind (N) in WORLD frame"""
        v_wind = self.wind_velocity_field(pos_world, t)
        F_drag = self.get_drag_force(vel_world, v_wind)
        F_momentum = self.get_momentum_disturbance()
        
        return F_drag + F_momentum


# ---------- Crazyflie Controller ----------

class CrazyflieController:
    """PyBullet-based Crazyflie simulation with wind disturbances"""
    def __init__(self, urdf_path="./cf2x.urdf", wind: WindDisturbance = None):
        self.mass = 0.027
        self.arm_length = 0.0397
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.max_rpm = 21000
        omega_max = self.max_rpm * 2*np.pi / 60.0
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)
        self.wind = wind
        self._wall_time = 0.0
        self._sim_dt = 1/20.0
        self.setup_pybullet(urdf_path)

    def quaternion2rotation_matrix(self, quaternion):
        """Convert quaternion [x,y,z,w] to rotation matrix"""
        n = np.dot(quaternion, quaternion)
        if n < 1e-12:
            return np.identity(3)
        q = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        q *= math.sqrt(2.0/n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2,2]-q[3,3], q[1,2]-q[3,0], q[1,3]+q[2,0]],
            [q[1,2]+q[3,0], 1.0-q[1,1]-q[3,3], q[2,3]-q[1,0]],
            [q[1,3]-q[2,0], q[2,3]+q[1,0], 1.0-q[1,1]-q[2,2]]
        ])

    def setup_pybullet(self, urdf_path):
        """Initialize PyBullet simulation"""
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        dt = self._sim_dt
        p.setTimeStep(dt)
        p.setPhysicsEngineParameter(numSubSteps=10)
        p.setRealTimeSimulation(0)
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF("plane.urdf")
        self.cf_id = p.loadURDF(urdf_path, start_pos, start_orientation, flags=p.URDF_USE_INERTIA_FROM_FILE)

    def mpc_to_motor_forces(self, thrust, moment_x, moment_y, moment_z):
        """Convert MPC outputs to motor forces"""
        c = self.km / self.kf
        control_vector = np.array([thrust, moment_x, moment_y, moment_z])
        d = self.arm_length / math.sqrt(2.0)
        mat = np.array([
            [1, 1, 1, 1],
            [ d, -d,  d, -d],
            [-d, -d,  d,  d],
            [c, -c, -c,  c]
        ])
        motor_forces = np.linalg.solve(mat, control_vector)
        return motor_forces

    def apply_motor_forces(self, motor_forces):
        """Apply motor forces and wind disturbances"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, _ang_vel = p.getBaseVelocity(self.cf_id)
        
        orn_array = np.array(orn)
        orn_normalized = orn_array / np.linalg.norm(orn_array)
        R_actual = self.quaternion2rotation_matrix(orn_normalized)
        
        # Apply motor forces
        d = self.arm_length / math.sqrt(2.0)
        offsets = np.array([
            [ d,  d, 0.],
            [ d, -d, 0.],
            [-d,  d, 0.],
            [-d, -d, 0.]
        ])
        
        for i in range(4):
            F_world = float(motor_forces[i]) * R_actual[:, 2]
            world_pos = (R_actual @ offsets[i]) + np.array(pos)
            p.applyExternalForce(self.cf_id, -1, F_world.tolist(), 
                               world_pos.tolist(), p.WORLD_FRAME)
        
        # Apply yaw torque
        c = self.km / self.kf
        tau_z_body = c * (motor_forces[0] - motor_forces[1] - motor_forces[2] + motor_forces[3])
        tau_world = R_actual[:, 2] * tau_z_body
        p.applyExternalTorque(self.cf_id, -1, tau_world.tolist(), p.WORLD_FRAME)
        
        # Apply wind disturbances
        if self.wind is not None:
            t_now = self._wall_time
            pos_w = np.array(pos)
            vel_w = np.array(vel)
            F_total = self.wind.get_total_force(t_now, self._sim_dt, pos_w, vel_w)
            p.applyExternalForce(self.cf_id, -1, F_total.tolist(), pos_w.tolist(), p.WORLD_FRAME)

    def get_state(self):
        """Get current drone state"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, ang_vel = p.getBaseVelocity(self.cf_id)
        
        orn_normalized = np.array(orn) / np.linalg.norm(orn)
        R = self.quaternion2rotation_matrix(orn_normalized)
        ang_vel_body = R.T @ np.array(ang_vel)
        
        return {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': np.array([orn_normalized[3], orn_normalized[0], orn_normalized[1], orn_normalized[2]]),
            'angular_velocity': ang_vel_body
        }

    def step_simulation(self, mpc_output, wall_time):
        """Execute one simulation step"""
        self._wall_time = wall_time
        motor_forces = self.mpc_to_motor_forces(
            mpc_output['thrust'], 
            mpc_output['moment_x'], 
            mpc_output['moment_y'], 
            mpc_output['moment_z']
        )
        self.apply_motor_forces(motor_forces)
        p.stepSimulation()
        return self.get_state()


# ---------- MPC ----------

def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion to Euler angles"""
    sinr_cosp = 2*(qw*qx + qy*qz)
    cosr_cosp = 1 - 2*(qx*qx + qy*qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2*(qw*qy - qz*qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2*(qw*qz + qx*qy)
    cosy_cosp = 1 - 2*(qy*qy + qz*qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


class MPC:
    """MPC """
    def __init__(self, target_height=1.0, wind: WindDisturbance = None):
        self.dt = 0.05
        self.N = 20
        self.m = 0.027
        self.g = 9.81
        self.target_height = target_height
        self.wind = wind
        
        self.T_ascend = 10.0
        self.T_hold = 10.0
        self.T_total = self.T_ascend + self.T_hold
        self.sim_steps = int(self.T_total / self.dt)
        
        self.model = quadrotor_model_auto()
        self.nx = int(self.model.x.size()[0])
        self.nu = int(self.model.u.size()[0])
        
        # Build trajectory with acceleration
        self.trajectory1 = self.build_piecewise_traj()
        self.simX1 = np.zeros((self.sim_steps + 1, self.nx))
        self.simU1 = np.zeros((self.sim_steps, self.nu))
        self.solve_times = np.zeros(self.sim_steps)
        self.errors = np.zeros((self.sim_steps, 3))
        
        self.u_prev = np.array([self.m * self.g, 0.0, 0.0, 0.0])
        
        # Integral action for disturbance rejection
        self.alt_i = 0.0
        self.alt_i_gain = 0.8
        self.alt_i_limit = 0.12
        self.xy_i = np.zeros(2)
        self.xy_i_gain = 0.8
        self.xy_i_limit = 0.15
        
        self.ocp = self.create_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        print(f"MPC initialized: N={self.N}, dt={self.dt}s, horizon={self.N*self.dt}s")
        

    def build_piecewise_traj(self):
        """Build trajectory with acceleration """
        nx = self.nx
        dt = self.dt
        
        # Ascent phase
        ascend = self.make_trajectory('quintic', {
            't': [0.0, self.T_ascend],
            'q': [[0.0, 0.0, 0.0], [0.0, 0.0, self.target_height]],
            'v': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'a': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'dt': dt
        })
        
        # Hold phase
        t_hold = np.arange(0.0, self.T_hold + dt, dt)
        N_hold = len(t_hold)
        q_hold = np.zeros((N_hold, 3))
        q_hold[:, 2] = self.target_height
        v_hold = np.zeros((N_hold, 3))
        a_hold = np.zeros((N_hold, 3))
        
        # Concatenate
        t_asc = ascend['t']
        q_asc = ascend['q']
        v_asc = ascend['v']
        a_asc = ascend['a']
        
        t_full = np.concatenate([t_asc, self.T_ascend + t_hold[1:]])
        q_full = np.vstack([q_asc, q_hold[1:]])
        v_full = np.vstack([v_asc, v_hold[1:]])
        a_full = np.vstack([a_asc, a_hold[1:]])
        
        # Build state trajectory with acceleration
        N_traj = len(t_full)
        traj = np.zeros((N_traj, nx + 3))  # Extended for acceleration
        traj[:, 0:3] = q_full    # position
        traj[:, 3:6] = v_full    # velocity
        traj[:, 6] = 1.0         # qw
        
        
        total_len = self.sim_steps + 1
        if N_traj < total_len:
            traj = np.vstack([traj, np.tile(traj[-1], (total_len - N_traj, 1))])
        
        return traj

    def create_ocp(self):
        """Create OCP """
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.N
        nx, nu = self.nx, self.nu
        
        # State indices
        IDX_R = slice(0, 3)
        IDX_V = slice(3, 6)
        IDX_Q = slice(6, 10)
        IDX_W = slice(10, 13)
        
        # Document-compliant cost weights
        Qr   = np.diag([80.0, 80.0, 150.0])      # Position error
        Qv   = np.diag([20.0, 20.0, 40.0])       # Velocity derivative (acceleration) error
        Qpsi = np.diag([12.0, 8.0, 8.0, 8.0])    # Attitude error
        Qw   = np.diag([6.0, 6.0, 5.0])          # Angular velocity error
        R    = np.diag([10, 10, 10, 10]) # Control deviation from u0
        
        # Terminal weights
        Pr   = 15.0 * Qr
        Ppsi = 15.0 * Qpsi
        
        # Stage residual y = [r - r_des; v - v_des; ψ - ψ_des; ω - ω_des; u - u0]
        n_res = 3 + 3 + 4 + 3 + nu
        W = np.zeros((n_res, n_res))
        W[0:3, 0:3]           = Qr
        W[3:6, 3:6]           = Qv
        W[6:10, 6:10]         = Qpsi
        W[10:13, 10:13]       = Qw
        W[13:13+nu, 13:13+nu] = R
        
        ocp.cost.cost_type   = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'
        ocp.cost.W = W
        
        # Terminal residual y_e = [r - r_des; ψ - ψ_des]
        n_res_e = 3 + 4
        W_e = np.zeros((n_res_e, n_res_e))
        W_e[0:3, 0:3] = Pr
        W_e[3:7, 3:7] = Ppsi
        ocp.cost.W_e = W_e
        
        # Selectors
        Vx = np.zeros((n_res, nx))
        Vx[0:3,   IDX_R] = np.eye(3)
        Vx[3:6,   IDX_V] = np.eye(3)
        Vx[6:10,  IDX_Q] = np.eye(4)
        Vx[10:13, IDX_W] = np.eye(3)
        
        Vu = np.zeros((n_res, nu))
        Vu[13:13+nu, :] = np.eye(nu)
        
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        
        Vx_e = np.zeros((n_res_e, nx))
        Vx_e[0:3, IDX_R] = np.eye(3)
        Vx_e[3:7, IDX_Q] = np.eye(4)
        ocp.cost.Vx_e = Vx_e
        
        ocp.cost.yref   = np.zeros(n_res)
        ocp.cost.yref_e = np.zeros(n_res_e)
        
        # Constraints
        mu = 1.3
        L = 0.0397
        d = L / math.sqrt(2.0)
        T_min = 0.0
        T_max = 4.0 * mu
        tau_xy_max = 2.6 * d * mu
        tau_z_max = tau_xy_max * 0.5
        
        ocp.constraints.lbu = np.array([T_min, -tau_xy_max, -tau_xy_max, -tau_z_max])
        ocp.constraints.ubu = np.array([T_max,  tau_xy_max,  tau_xy_max,  tau_z_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        
        # State constraints
        vx_max = 1.5
        vy_max = 1.5
        max_angular_vel = 3.5
        max_position = 10.0
        
        ocp.constraints.lbx = np.array([
            -vx_max, -vy_max, 0.0, 
            -max_angular_vel, -max_angular_vel, -max_angular_vel
        ])
        ocp.constraints.ubx = np.array([
            vx_max, vy_max, max_position, 
            max_angular_vel, max_angular_vel, max_angular_vel
        ])
        ocp.constraints.idxbx = np.array([3, 4, 2, 10, 11, 12], dtype=int)
        
        # Input rate constraints
        du_max_thrust = 12.0 * self.dt
        du_max_torque = 1.5 * self.dt
        
        ocp.constraints.lbu_rate = np.array([
            -du_max_thrust, -du_max_torque, -du_max_torque, -du_max_torque*0.5
        ])
        ocp.constraints.ubu_rate = np.array([
            du_max_thrust, du_max_torque, du_max_torque, du_max_torque*0.5
        ])
        ocp.constraints.idxbu_rate = np.array([0, 1, 2, 3])
        
        ocp.constraints.x0 = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        
        # Solver options
        ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.tf = self.N * self.dt
        
        return ocp

    def solve(self, step, current_state):
        """Solve MPC with wind compensation"""
        mg = self.m * self.g
        
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)
        
        # Get reference
        ref_idx = min(step, len(self.trajectory1)-1)
        r_des = self.trajectory1[ref_idx, 0:3]
        
        # Integral action
        z_err = r_des[2] - current_state[2]
        self.alt_i += z_err * self.dt * self.alt_i_gain
        self.alt_i = float(np.clip(self.alt_i, -self.alt_i_limit, self.alt_i_limit))
        
        xy_err = r_des[0:2] - current_state[0:2]
        self.xy_i += xy_err * self.dt * self.xy_i_gain
        self.xy_i = np.clip(self.xy_i, -self.xy_i_limit, self.xy_i_limit)
        
        # Tilt-compensated thrust with wind feedforward
        qw, qx, qy, qz = current_state[6:10]
        roll, pitch, _ = quat_to_euler(qw, qx, qy, qz)
        tilt_cos = max(0.3, math.cos(roll) * math.cos(pitch))
        
        T_base = mg + self.alt_i
        
        # Wind compensation
        if self.wind is not None:
            t_now = step * self.dt
            pos_world = current_state[0:3]
            v_wind_est = self.wind.wind_velocity_field(pos_world, t_now)
            vel_world = current_state[3:6]
            
            v_rel = vel_world - v_wind_est
            speed_rel = np.linalg.norm(v_rel)
            if speed_rel > 0.1:
                drag_magnitude = (0.5 * self.wind.air_density * 
                                self.wind.drag_cd * self.wind.drag_area * speed_rel)
                drag_z = drag_magnitude * abs(v_rel[2]) / (speed_rel + 1e-6)
                T_base += 0.6 * drag_z
        
        T_req = T_base / tilt_cos
        T_max = self.ocp.constraints.ubu[0]
        T_req = float(np.clip(T_req, 0.0, T_max))
        u0 = np.array([T_req, 0.0, 0.0, 0.0])  # Nominal control
        
        # Set stage references with velocity
        for k in range(self.N):
            idx = min(step + k, len(self.trajectory1) - 1)
            xref = self.trajectory1[idx]
            r_des   = xref[0:3]
            v_des   = xref[3:6]  # Use velocity directly
            psi_des = xref[6:10]
            w_des   = np.zeros(3)
            
            # yref = [r_des; v_des; psi_des; w_des; u0]
            yref = np.hstack([r_des, v_des, psi_des, w_des, u0])
            self.solver.set(k, "yref", yref)
        
        # Terminal reference
        idx_T = min(step + self.N, len(self.trajectory1) - 1)
        xref_T = self.trajectory1[idx_T]
        yref_e = np.hstack([xref_T[0:3], xref_T[6:10]])
        self.solver.set(self.N, "yref", yref_e)
        
        # Solve
        t_start = time.time()
        status = self.solver.solve()
        solve_time = time.time() - t_start
        
        if status not in [0, 2]:
            print(f"Warning: Solver status {status} at step {step}")
        
        # Extract optimal control
        u_opt = self.solver.get(0, "u")
        self.u_prev = u_opt.copy()
        
        # Log data
        self.simX1[step] = current_state
        self.simU1[step] = u_opt
        self.solve_times[step] = solve_time
        
        # Track error
        pos_error = current_state[0:3] - self.trajectory1[ref_idx, 0:3]
        self.errors[step] = pos_error
        
        return {
            'thrust': float(u_opt[0]),
            'moment_x': float(u_opt[1]),
            'moment_y': float(u_opt[2]),
            'moment_z': float(u_opt[3])
        }

    def make_trajectory(self, traj_type, params):
        """Generate smooth polynomial trajectories"""
        t0, tf = params['t']
        q0, qf = np.array(params['q'][0]), np.array(params['q'][1])
        v0, vf = np.array(params['v'][0]), np.array(params['v'][1])
        dt = params['dt']
        D = len(q0)
        
        if traj_type == 'quintic':
            a0, af = np.array(params['a'][0]), np.array(params['a'][1])
        
        times = np.arange(t0, tf + dt, dt)
        N = len(times)
        q_traj = np.zeros((N, D))
        v_traj = np.zeros((N, D))
        a_traj = np.zeros((N, D))
        
        for d in range(D):
            if traj_type == 'quintic':
                A = np.array([
                    [1, t0, t0**2, t0**3, t0**4, t0**5],
                    [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                    [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                    [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]
                ])
                b = np.array([q0[d], v0[d], a0[d], qf[d], vf[d], af[d]])
                coeffs = np.linalg.solve(A, b)
                
                for i, t in enumerate(times):
                    q_traj[i, d] = (coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + 
                                   coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5)
                    v_traj[i, d] = (coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t**2 + 
                                   4*coeffs[4]*t**3 + 5*coeffs[5]*t**4)
                    a_traj[i, d] = (2*coeffs[2] + 6*coeffs[3]*t + 12*coeffs[4]*t**2 + 
                                   20*coeffs[5]*t**3)
        
        return {'t': times, 'q': q_traj, 'v': v_traj, 'a': a_traj}

    def report_results(self):
        """Print comprehensive performance statistics"""
        print("\n" + "="*70)
        print("NMPC ROBUSTNESS TEST RESULTS")
        print("="*70)
        
        pos_errors = np.linalg.norm(self.errors, axis=1)
        print(f"\nPosition Tracking Performance:")
        print(f"  Mean Error:     {np.mean(pos_errors):.4f} m")
        print(f"  Max Error:      {np.max(pos_errors):.4f} m")
        print(f"  RMS Error:      {np.sqrt(np.mean(pos_errors**2)):.4f} m")
        print(f"  Std Dev:        {np.std(pos_errors):.4f} m")
        
        print(f"\nAxis-wise Errors:")
        print(f"  X - Mean: {np.mean(np.abs(self.errors[:,0])):.4f} m, Max: {np.max(np.abs(self.errors[:,0])):.4f} m")
        print(f"  Y - Mean: {np.mean(np.abs(self.errors[:,1])):.4f} m, Max: {np.max(np.abs(self.errors[:,1])):.4f} m")
        print(f"  Z - Mean: {np.mean(np.abs(self.errors[:,2])):.4f} m, Max: {np.max(np.abs(self.errors[:,2])):.4f} m")
        
        print(f"\nControl Inputs:")
        print(f"  Thrust - Mean: {np.mean(self.simU1[:,0]):.4f} N, "
              f"Max: {np.max(self.simU1[:,0]):.4f} N, "
              f"Std: {np.std(self.simU1[:,0]):.4f} N")
        print(f"  Torque X - Mean: {np.mean(np.abs(self.simU1[:,1])):.6f} Nm, "
              f"Max: {np.max(np.abs(self.simU1[:,1])):.6f} Nm")
        print(f"  Torque Y - Mean: {np.mean(np.abs(self.simU1[:,2])):.6f} Nm, "
              f"Max: {np.max(np.abs(self.simU1[:,2])):.6f} Nm")
        print(f"  Torque Z - Mean: {np.mean(np.abs(self.simU1[:,3])):.6f} Nm, "
              f"Max: {np.max(np.abs(self.simU1[:,3])):.6f} Nm")
        
        print(f"\nSolver Performance:")
        print(f"  Mean solve time: {np.mean(self.solve_times)*1000:.2f} ms")
        print(f"  Max solve time:  {np.max(self.solve_times)*1000:.2f} ms")
        print(f"  Real-time capable: {np.max(self.solve_times) < self.dt}")
        print("="*70 + "\n")


# Main Simulation 

if __name__ == "__main__":
    print("="*50)
    print("NMPC WIND ROBUSTNESS TEST")
    print("="*50)
    
    # Configure turbulence
    turbulence = TurbulenceModel(
        sigma=np.array([0.2, 0.2, 0.1]),
        L=np.array([25.0, 25.0, 15.0]),
        airspeed_ref=2.0,
        seed=42
    )
    
    # Configure wind
    wind = WindDisturbance(
        mode="mixed",
        const_wind_vel=np.array([0.8, 0.0, 0.0]),
        sin_amp_vel=np.array([0.4, 0.2, 0.0]),
        sin_freq=0.2,
        gust_max_vel=np.array([1.5, 1.0, 0.8]),
        gust_pulse_dur=0.5,
        gust_cooldown=(2.0, 4.0),
        air_density=1.225,
        drag_cd=1.2,
        drag_area=0.010,
        turbulence=turbulence,
        enable_momentum_disturbances=False,
        momentum_disturbance_scale=0.01
    )
    wind.reset(0.0)
    
    print("\nWind Configuration:")
    print(f"  Mode: {wind.mode}")
    print(f"  Constant wind: {wind.const_wind_vel} m/s")
    print(f"  Sinusoidal amplitude: {wind.sin_amp_vel} m/s @ {wind.sin_freq} Hz")
    print(f"  Random gust max: {wind.gust_max_vel} m/s")
    print(f"  Turbulence RMS: {turbulence.sigma} m/s")
    
    # Initialize
    cf_controller = CrazyflieController(wind=wind)
    mpc = MPC(target_height=1.0, wind=wind)
    
    print(f"\nStarting simulation for {mpc.sim_steps} steps ({mpc.T_total}s)...")
    print(f"Target altitude: {mpc.target_height} m")
    print(f"Cost: Document-compliant (acceleration tracking)\n")
    
    start_wall = time.time()
    
    for step in range(mpc.sim_steps):
        state = cf_controller.get_state()
        x = np.concatenate([
            state['position'].ravel(),
            state['velocity'].ravel(),
            state['orientation'].ravel(),
            state['angular_velocity'].ravel()
        ])
        
        u = mpc.solve(step, x)
        
        wall_time = time.time() - start_wall
        cf_controller.step_simulation(u, wall_time)
        
        if step % 40 == 0:
            pos = x[0:3]
            vel = x[3:6]
            ref_pos = mpc.trajectory1[step, 0:3]
            err = np.linalg.norm(pos - ref_pos)
            
            v_wind = wind.wind_velocity_field(pos, wall_time)
            wind_mag = np.linalg.norm(v_wind)
            
            print(f"Step {step:4d}/{mpc.sim_steps} | "
                  f"Pos=[{pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}]m | "
                  f"Vel=[{vel[0]:5.2f},{vel[1]:5.2f},{vel[2]:5.2f}]m/s | "
                  f"Err={err:.3f}m | Wind={wind_mag:.2f}m/s")
        
        next_tick = start_wall + (step + 1) * mpc.dt
        remaining = next_tick - time.time()
        if remaining > 0:
            time.sleep(remaining)
    
    # Final state
    state = cf_controller.get_state()
    x = np.concatenate([
        state['position'].ravel(),
        state['velocity'].ravel(),
        state['orientation'].ravel(),
        state['angular_velocity'].ravel()
    ])
    mpc.simX1[mpc.sim_steps] = x
    
    mpc.report_results()
    
    # Visualization
    traj_plot1 = mpc.trajectory1[:mpc.sim_steps+1, :3]
    fig = plt.figure(figsize=(14, 12))
    
    # Position tracking
    ax1 = fig.add_subplot(3, 2, 1)
    times_full = np.arange(mpc.sim_steps+1) * mpc.dt
    ax1.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 0], 'r-', linewidth=2, label='x')
    ax1.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 1], 'g-', linewidth=2, label='y')
    ax1.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 2], 'b-', linewidth=2, label='z')
    ax1.plot(times_full, traj_plot1[:, 0], 'r--', alpha=0.5, label='x ref')
    ax1.plot(times_full, traj_plot1[:, 1], 'g--', alpha=0.5, label='y ref')
    ax1.plot(times_full, traj_plot1[:, 2], 'b--', alpha=0.5, label='z ref')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Position [m]')
    ax1.set_title('Position Tracking')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Tracking error
    ax2 = fig.add_subplot(3, 2, 2)
    times = np.arange(mpc.sim_steps) * mpc.dt
    ax2.plot(times, np.linalg.norm(mpc.errors, axis=1), 'b-', linewidth=2)
    ax2.fill_between(times, 0, np.linalg.norm(mpc.errors, axis=1), alpha=0.3)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Error [m]')
    ax2.set_title('Tracking Error')
    ax2.grid(True, alpha=0.3)
    
    # Velocity
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 3], 'r-', linewidth=2, label='vx')
    ax3.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 4], 'g-', linewidth=2, label='vy')
    ax3.plot(times_full, mpc.simX1[:mpc.sim_steps+1, 5], 'b-', linewidth=2, label='vz')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Velocity [m/s]')
    ax3.set_title('Velocity Components')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Thrust
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.plot(times, mpc.simU1[:, 0], 'b-', linewidth=2, label='Thrust')
    ax4.axhline(y=mpc.m*mpc.g, color='r', linestyle='--', label='Hover')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Thrust [N]')
    ax4.set_title('Control: Thrust')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Torques
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(times, mpc.simU1[:, 1], linewidth=2, label='τ_x')
    ax5.plot(times, mpc.simU1[:, 2], linewidth=2, label='τ_y')
    ax5.plot(times, mpc.simU1[:, 3], linewidth=2, label='τ_z')
    ax5.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax5.set_xlabel('Time [s]')
    ax5.set_ylabel('Torque [Nm]')
    ax5.set_title('Control: Torques')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Solve times
    ax6 = fig.add_subplot(3, 2, 6)
    solve_times_ms = mpc.solve_times * 1000
    ax6.plot(times, solve_times_ms, 'g-', linewidth=1.5)
    ax6.axhline(y=mpc.dt*1000, color='r', linestyle='--', linewidth=2, 
                label=f'RT limit ({mpc.dt*1000:.0f}ms)')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Solve Time [ms]')
    ax6.set_title('MPC Solver Performance')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('wind_robustness.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'wind_robustness.png'")
    plt.show()
    
    p.disconnect()
    print("\nSimulation complete!")
