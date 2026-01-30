from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos

def quadrotor_model_auto() -> AcadosModel:
    """
    Create a quadrotor model for acados MPC
    
    State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]
    - Position: (x, y, z)
    - Velocity: (vx, vy, vz) 
    - Attitude: (qw, qx, qy, qz)
    - Angular rates: (p, q, r)
    
    Control vector: [u1, tau_x, tau_y, tau_z]
    - u1: total thrust
    - tau_x, tau_y, tau_z: torques around body axes
    """
    
    model_name = 'quadrotor'
    
    # Parameters from Table 1
    m = 0.027      # kg
    g = 9.81       # m/s^2
    I11 = 1.4e-5   # kg*m^2
    I22 = 1.4e-5   # kg*m^2
    I33 = 2.17e-5  # kg*m^2
    
    # State variables
    x = SX.sym('x')
    y = SX.sym('y')
    z = SX.sym('z')
    vx = SX.sym('vx')
    vy = SX.sym('vy')
    vz = SX.sym('vz')
    qw = SX.sym('qw')
    qx = SX.sym('qx')
    qy = SX.sym('qy')
    qz = SX.sym('qz')
    p = SX.sym('p')
    q = SX.sym('q')
    r = SX.sym('r')
    
    state = vertcat(x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r)
    
    # Control inputs
    u1 = SX.sym('u1')
    tau_x = SX.sym('tau_x')
    tau_y = SX.sym('tau_y')
    tau_z = SX.sym('tau_z')
    
    controls = vertcat(u1, tau_x, tau_y, tau_z)
    
    # State derivatives
    x_dot = SX.sym('x_dot')
    y_dot = SX.sym('y_dot')
    z_dot = SX.sym('z_dot')
    vx_dot = SX.sym('vx_dot')
    vy_dot = SX.sym('vy_dot')
    vz_dot = SX.sym('vz_dot')
    qw_dot = SX.sym('qw_dot')
    qx_dot = SX.sym('qx_dot')
    qy_dot = SX.sym('qy_dot')
    qz_dot = SX.sym('qz_dot')
    p_dot = SX.sym('p_dot')
    q_dot = SX.sym('q_dot')
    r_dot = SX.sym('r_dot')
    
    xdot = vertcat(x_dot, y_dot, z_dot, vx_dot, vy_dot, vz_dot,
                   qw_dot, qx_dot, qy_dot, qz_dot, p_dot, q_dot, r_dot)
    
    # Rotation matrix from quaternion (body to world)
    R11 = 1 - 2*(qy**2 + qz**2)
    R12 = 2*(qx*qy - qw*qz)
    R13 = 2*(qx*qz + qw*qy)
    R21 = 2*(qx*qy + qw*qz)
    R22 = 1 - 2*(qx**2 + qz**2)
    R23 = 2*(qy*qz - qw*qx)
    R31 = 2*(qx*qz - qw*qy)
    R32 = 2*(qy*qz + qw*qx)
    R33 = 1 - 2*(qx**2 + qy**2)
    
    # Thrust in world frame
    thrust_x = (R13 * u1) / m
    thrust_y = (R23 * u1) / m
    thrust_z = (R33 * u1) / m
    
    # Dynamics equations
    f_expl = vertcat(
        vx,                                          # x_dot
        vy,                                          # y_dot
        vz,                                          # z_dot
        thrust_x,                                    # vx_dot
        thrust_y,                                    # vy_dot
        thrust_z - g,                                # vz_dot
        0.5 * (-p*qx - q*qy - r*qz),                # qw_dot
        0.5 * (p*qw + r*qy - q*qz),                 # qx_dot
        0.5 * (q*qw - r*qx + p*qz),                 # qy_dot
        0.5 * (r*qw + q*qx - p*qy),                 # qz_dot
        (tau_x - (I33 - I22)*q*r) / I11,            # p_dot
        (tau_y - (I11 - I33)*p*r) / I22,            # q_dot
        (tau_z - (I22 - I11)*p*q) / I33             # r_dot
    )
    
    f_impl = xdot - f_expl
    
    # Create model
    model = AcadosModel()
    
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = state
    model.xdot = xdot
    model.u = controls
    model.name = model_name
    
    return model
