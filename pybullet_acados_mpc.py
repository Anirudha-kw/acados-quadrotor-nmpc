import pybullet as p
import pybullet_data
import numpy as np
import time
import math

from acados_template import AcadosOcp, AcadosOcpSolver
from quadrotor_model import quadrotor_model_auto
import matplotlib.pyplot as plt

class CrazyflieController:
    def __init__(self, urdf_path="./cf2x.urdf"):
        self.mass = 0.027
        self.arm_length = 0.0397
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.max_rpm = 21000
        omega_max = self.max_rpm * 2*np.pi / 60.0
        self.max_thrust_per_motor = self.kf * (omega_max ** 2)
        self.setup_pybullet(urdf_path)
    
    def quaternion2rotation_matrix(self, quaternion):
        """quaternion: [x,y,z,w] (PyBullet format) to rotation matrix"""
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
        # Start video recording
        p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "quadrotor_simulation.mp4")

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        dt = 1/20.0
        p.setTimeStep(dt)
        p.setPhysicsEngineParameter(numSubSteps=20)
        p.setRealTimeSimulation(0)
        
        start_pos = [0, 0, 0]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.loadURDF("plane.urdf")
        self.cf_id = p.loadURDF(urdf_path, start_pos, start_orientation, 
                                flags=p.URDF_USE_INERTIA_FROM_FILE)
        
        self.motor_links = []
        for i in range(4):
            link_name = f"prop{i}_link"
            for j in range(p.getNumJoints(self.cf_id)):
                link_info = p.getJointInfo(self.cf_id, j)
                if link_info[12].decode('utf-8') == link_name:
                    self.motor_links.append(j)
                    break
        # p.changeDynamics(bodyUniqueId=self.cf_id, linkIndex=-1, lateralFriction=0.0)
        print(f"PyBullet Dynamics: {p.getDynamicsInfo(self.cf_id, -1)}")
        print(f"Found motor links at indices: {self.motor_links}")

    def mpc_to_motor_forces(self, thrust, moment_x, moment_y, moment_z):
        """Convert MPC outputs to motor forces"""
        c = self.km / self.kf
        control_vector = np.array([thrust, moment_x, moment_y, moment_z])
        d = self.arm_length / math.sqrt(2.0)
        
        mat = np.array([
            [1, 1, 1, 1], 
            [d, -d, d, -d], 
            [-d, -d, d, d],
            [c, -c, -c, c]
        ])
        
        motor_forces = np.linalg.inv(mat) @ control_vector
        mu = 0.73575
        motor_forces = np.clip(motor_forces, 0.0, mu)
        return motor_forces

    def apply_motor_forces(self, motor_forces):
        """Apply thrust forces at the four rotors"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        orn_array = np.array(orn)
        orn_normalized = orn_array / np.linalg.norm(orn_array)
        R_actual = self.quaternion2rotation_matrix(orn_normalized)
        
        d = self.arm_length / math.sqrt(2.0)
        offsets = np.array([
            [d, d, 0.0],
            [d, -d, 0.0],
            [-d, d, 0.0],
            [-d, -d, 0.0]
        ])
        
        for i in range(4):
            F_world = float(motor_forces[i]) * R_actual[:, 2]
            world_pos = (R_actual @ offsets[i]) + np.array(pos)
            p.applyExternalForce(self.cf_id, -1, F_world.tolist(), 
                               world_pos.tolist(), p.WORLD_FRAME)
        
        c = self.km / self.kf
        tau_z_body = c * (motor_forces[0] - motor_forces[1] - motor_forces[2] + motor_forces[3])
        tau_world = R_actual[:, 2] * tau_z_body
        p.applyExternalTorque(self.cf_id, -1, tau_world.tolist(), p.WORLD_FRAME)

    def get_state(self):
        """Get current state for MPC feedback"""
        pos, orn = p.getBasePositionAndOrientation(self.cf_id)
        vel, ang_vel = p.getBaseVelocity(self.cf_id)
        
        orn_array = np.array(orn)
        orn_normalized = orn_array / np.linalg.norm(orn_array)
        
        # Convert to [qw, qx, qy, qz] format
        quat_state = np.array([orn_normalized[3], orn_normalized[0], 
                              orn_normalized[1], orn_normalized[2]])
        
        # Convert angular velocity to body frame
        R = self.quaternion2rotation_matrix(orn_normalized)
        ang_vel_body = R.T @ np.array(ang_vel)
        
        return {
            'position': np.array(pos),
            'velocity': np.array(vel),
            'orientation': quat_state,
            'angular_velocity': ang_vel_body
        }
    
    def step_simulation(self, mpc_output):
        """One simulation step with MPC control"""
        motor_forces = self.mpc_to_motor_forces(
            mpc_output['thrust'],
            mpc_output['moment_x'], 
            mpc_output['moment_y'],
            mpc_output['moment_z']
        )
        
        self.apply_motor_forces(motor_forces)
        p.stepSimulation()
        
        return self.get_state()


class MPC:
    """NMPC for the drone"""
    def __init__(self, target_height=1.0):
        self.dt = 0.05
        self.N = 20
        self.m = 0.027
        self.g = 9.81
        self.target_height = target_height
        
        self.T_total = 10.0
        self.sim_steps = int(self.T_total / self.dt)
        
        self.model = quadrotor_model_auto()
        self.nx = int(self.model.x.size()[0])
        self.nu = int(self.model.u.size()[0])
        
        # Generate trajectory from (0,0,0) to (1,1,target_height)
        traj_params = {
            't': [0, self.T_total],
            'q': [[0.0, 0.0, 0.0], [1.0, 1.0, self.target_height]],
            'v': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'a': [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            'dt': self.dt
        }
        
        traj = self.make_trajectory('quintic', traj_params)
        
        # Build full state trajectory 
        N_traj = len(traj['t'])
        self.trajectory1 = np.zeros((N_traj, self.nx))  # No extra columns needed
        self.trajectory1[:, 0:3] = traj['q']      # Position
        self.trajectory1[:, 3:6] = traj['v']      # Velocity (ṙ_des)
        self.trajectory1[:, 6] = 1.0              # qw = 1
        
        # Extend if needed
        total_len = self.sim_steps + 1
        if N_traj < total_len:
            self.trajectory1 = np.vstack([
                self.trajectory1, 
                np.tile(self.trajectory1[-1], (total_len - N_traj, 1))
            ])
        
        # Storage 
        self.simX1 = np.zeros((self.sim_steps + 1, self.nx))
        self.simU1 = np.zeros((self.sim_steps, self.nu))
        self.solve_times = np.zeros(self.sim_steps)
        self.errors = np.zeros((self.sim_steps, 3))
        self.u_prev = np.array([self.m * self.g, 0.0, 0.0, 0.0])
        
        # Create OCP and solver
        self.ocp = self.create_ocp()
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')
        
        print(f"MPC initialized: N={self.N}, dt={self.dt}s, T={self.N*self.dt}s")
        print(f"Target: (0,0,0) → (1,1,{self.target_height})")
        print(f"Velocity tracking formulation")
    
    def make_trajectory(self, traj_type, params):
        """Multi-dimensional quintic trajectory generator"""
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
                    a_traj[i, d] = (2*coeffs[2] + 6*coeffs[3]*t +
                                    12*coeffs[4]*t**2 + 20*coeffs[5]*t**3)

        return {'t': times, 'q': q_traj, 'v': v_traj, 'a': a_traj}
    
    def create_ocp(self):
        """Create OCP"""
        ocp = AcadosOcp()
        ocp.model = self.model
        ocp.dims.N = self.N
        nx, nu = self.nx, self.nu
        
        # State indices
        IDX_R = slice(0, 3)
        IDX_V = slice(3, 6)
        IDX_Q = slice(6, 10)
        IDX_W = slice(10, 13)
        
        # Cost weights 
        Qr   = np.diag([200.0, 200.0, 300.0])     # Position error
        Qv   = np.diag([30.0, 30.0, 50.0])        # Velocity tracking error
        Qpsi = np.diag([1.0, 1.0, 1.0, 1.0])      # Attitude error
        Qw   = np.diag([1.0, 1.0, 0.5])           # Angular velocity error
        R    = np.diag([0.01, 0.01, 0.01, 0.01])  # Control deviation from u0
        
        # Terminal weights
        Pr   = 20000.0 * np.eye(3)  # Strong terminal position
        Ppsi = 200.0 * np.eye(4)    # Terminal attitude
        
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
        mu = 0.73575
        L = 0.0397
        d = L / math.sqrt(2.0)
        
        T_min = 0.0
        T_max = 4.0 * mu
        tau_xy_max = 2.0 * d * mu
        tau_z_max = tau_xy_max * 0.5
        
        print(f"Torque constraints: τ_xy_max = {tau_xy_max:.6f} Nm, τ_z_max = {tau_z_max:.6f} Nm")
        
        ocp.constraints.lbu = np.array([T_min, -tau_xy_max, -tau_xy_max, -tau_z_max])
        ocp.constraints.ubu = np.array([T_max,  tau_xy_max,  tau_xy_max,  tau_z_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        
        # State constraints
        max_angular_vel = 3.0
        max_z = 10.0
        
        ocp.constraints.lbx = np.array([0.0, -max_angular_vel, -max_angular_vel, -max_angular_vel])
        ocp.constraints.ubx = np.array([max_z, max_angular_vel, max_angular_vel, max_angular_vel])
        ocp.constraints.idxbx = np.array([2, 10, 11, 12])
        
        # Input rate constraints
        du_max_thrust = 12.0 * self.dt
        du_max_torque = 1.2 * self.dt
        
        ocp.constraints.lbu_rate = np.array([-du_max_thrust, -du_max_torque, -du_max_torque, -du_max_torque*0.6])
        ocp.constraints.ubu_rate = np.array([du_max_thrust, du_max_torque, du_max_torque, du_max_torque*0.6])
        ocp.constraints.idxbu_rate = np.array([0, 1, 2, 3])
        
        # Initial state
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
        """Solve MPC"""
        hover_thrust = self.m * self.g
        u0 = np.array([hover_thrust, 0.0, 0.0, 0.0])
        
        # Update initial condition
        self.solver.set(0, "lbx", current_state)
        self.solver.set(0, "ubx", current_state)
        
        # Update reference trajectory
        for k in range(self.N):
            idx = min(step + k, len(self.trajectory1) - 1)
            xref = self.trajectory1[idx]
            r_des = xref[0:3]          # Position reference
            v_des = xref[3:6]          # Velocity reference (ṙ_des)
            psi_des = xref[6:10]       # Quaternion reference
            w_des = np.zeros(3)        # Zero angular velocity
            
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
        
        if status != 0:
            print(f"Warning: Solver status {status} at step {step}")
        
        # Extract optimal control
        u_opt = self.solver.get(0, "u")
        
        # Debug output
        if step % 40 == 0:
            print(f"\n=== Step {step} ===")
            print(f"Current pos: {current_state[0:3]}")
            print(f"Target pos: {self.trajectory1[step, 0:3]}")
            print(f"Position error: {np.linalg.norm(current_state[0:3] - self.trajectory1[step, 0:3]):.6f}")
            print(f"Control: T={u_opt[0]:.4f}, τ=[{u_opt[1]:.6f},{u_opt[2]:.6f},{u_opt[3]:.6f}]")
        
        self.u_prev = u_opt.copy()
        
        # Store data
        self.simX1[step] = current_state
        self.simU1[step] = u_opt
        self.solve_times[step] = solve_time
        
        # Compute error
        ref_idx = min(step, len(self.trajectory1) - 1)
        pos_error = current_state[0:3] - self.trajectory1[ref_idx, 0:3]
        self.errors[step] = pos_error
        
        return {
            'thrust': float(u_opt[0]),
            'moment_x': float(u_opt[1]),
            'moment_y': float(u_opt[2]),
            'moment_z': float(u_opt[3])
        }
    
    def report_results(self):
        """Report results"""
        print("\n" + "="*60)
        print("PHASE 3: FROM (0,0,0) TO (1,1,1) TRAJECTORY TRACKING")
        print("="*60)
        print(f"Trajectory: (0,0,0) → (1,1,{self.target_height})")
        print(f"Duration: {self.T_total}s, Steps: {self.sim_steps}")
        
        pos_errors = np.linalg.norm(self.errors, axis=1)
        print(f"\nPosition Tracking Error:")
        print(f"  Mean: {np.mean(pos_errors):.4f} m")
        print(f"  Max:  {np.max(pos_errors):.4f} m")
        print(f"  RMS:  {np.sqrt(np.mean(pos_errors**2)):.4f} m")
        print(f"  Final: {pos_errors[-1]:.4f} m")
        
        print(f"\nControl Inputs:")
        print(f"  Thrust - Mean: {np.mean(self.simU1[:,0]):.4f} N, "
              f"Max: {np.max(self.simU1[:,0]):.4f} N")
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
        
        print("="*60 + "\n")

# Main simulation loop
if __name__ == "__main__":
    cf_controller = CrazyflieController()
    mpc = MPC(target_height=1.0)
    
    print(f"Starting simulation for {mpc.sim_steps} steps...")
    
    for step in range(mpc.sim_steps):
        current_state_dict = cf_controller.get_state()
        current_state = np.concatenate([
            current_state_dict['position'].ravel(),
            current_state_dict['velocity'].ravel(),
            current_state_dict['orientation'].ravel(),
            current_state_dict['angular_velocity'].ravel()
        ])
        
        mpc_output = mpc.solve(step, current_state)
        cf_controller.step_simulation(mpc_output)

        
        
        if step % 20 == 0:
            pos = current_state[0:3]
            ref_pos = mpc.trajectory1[step, 0:3]
            error = np.linalg.norm(pos - ref_pos)
            print(f"Step {step}/{mpc.sim_steps}: "
                  f"Pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}], "
                  f"Error={error:.4f}m")
    
    # Store final state
    current_state_dict = cf_controller.get_state()
    current_state = np.concatenate([
        current_state_dict['position'].ravel(),
        current_state_dict['velocity'].ravel(),
        current_state_dict['orientation'].ravel(),
        current_state_dict['angular_velocity'].ravel()
    ])
    mpc.simX1[mpc.sim_steps] = current_state
    
    mpc.report_results()
    
    # Plotting
    traj_plot1 = mpc.trajectory1[:mpc.sim_steps+1, :3]
    
    fig = plt.figure(figsize=(12, 10))
    
    # 3D trajectory
    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    
    ax.plot(traj_plot1[:, 0], traj_plot1[:, 1], traj_plot1[:, 2], 
            '--', label='Reference', color='r', linewidth=2)
    ax.plot(mpc.simX1[:,0], mpc.simX1[:,1], mpc.simX1[:,2], 
            'b-', label='Actual', linewidth=1.5)
    ax.scatter([0], [0], [0], c='g', marker='o', s=100, label='Start')
    ax.scatter([1], [1], [1], c='r', marker='x', s=100, label='Goal')
    ax.legend()
    ax.grid(True)
    
    # Tracking error
    ax2 = fig.add_subplot(222)
    times = np.arange(mpc.sim_steps) * mpc.dt
    ax2.plot(times, np.linalg.norm(mpc.errors, axis=1), 'b-', linewidth=1.5)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position Error [m]')
    ax2.set_title('Tracking Error')
    ax2.grid(True)
    
    # Thrust
    ax3 = fig.add_subplot(223)
    ax3.plot(times, mpc.simU1[:, 0], 'b-', linewidth=1.5, label='Thrust')
    ax3.axhline(y=mpc.m*mpc.g, color='r', linestyle='--', label='Hover', alpha=0.5)
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Thrust [N]')
    ax3.set_title('Control Input: Thrust')
    ax3.grid(True)
    ax3.legend()
    
    # Torques
    ax4 = fig.add_subplot(224)
    ax4.plot(times, mpc.simU1[:, 1], label='τ_x (roll)', linewidth=1.5)
    ax4.plot(times, mpc.simU1[:, 2], label='τ_y (pitch)', linewidth=1.5)
    ax4.plot(times, mpc.simU1[:, 3], label='τ_z (yaw)', linewidth=1.5)
    ax4.set_ylabel('Torque [Nm]')
    ax4.set_title('Control Inputs: Torques')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('phase3_trajectory_tracking.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'phase3_trajectory_tracking_velocity.png'")
    
    p.disconnect()
    print("\nSimulation complete!")
