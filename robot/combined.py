import os
import math
import numpy as np
import scipy.linalg
from scipy.interpolate import CubicSpline
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt

xml_path = "ip2d.xml"
simend = 15.0
move_start = 0.5
auto_test = True
lock_steer = False
direct_steer_control = True  # 直接控制舵角（跳过PID）

vx_cmd = 0.0
vy_cmd = 0.0
wz_cmd = 0.0
v_step = 0.1
w_step = 0.3
v_max = 3.0
w_max = 3.0


class PIDController:
    """PID控制器类"""
    def __init__(self, kp, ki, kd, int_min, int_max, out_min, out_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.int_min = int_min
        self.int_max = int_max
        self.out_min = out_min
        self.out_max = out_max
        self.integral = 0.0
    
    def compute(self, error, derivative, dt):
        self.integral = clamp(self.integral + error * dt, self.int_min, self.int_max)
        output = self.kp * error + self.ki * self.integral - self.kd * derivative
        return clamp(output, self.out_min, self.out_max)
    
    def reset(self):
        self.integral = 0.0


pid_vx = PIDController(kp=60.0, ki=0.0, kd=0.0, int_min=-0.0, int_max=0.0, out_min=-80.0, out_max=80.0)
pid_wz = PIDController(kp=10.0, ki=10.0, kd=100.0, int_min=-10.0, int_max=10.0, out_min=-2.0, out_max=2.0) # todo：积分分离
# pid_steer_l = PIDController(kp=40.0, ki=0.5, kd=2.0, int_min=-1.0, int_max=1.0, out_min=-50.0, out_max=50.0)
# pid_steer_r = PIDController(kp=40.0, ki=0.5, kd=2.0, int_min=-1.0, int_max=1.0, out_min=-50.0, out_max=50.0)

K_interpolator = None
x0 = None
qpos0 = None
qvel0 = None
Q = np.diag([200000.0, 6000000.0, 5000000.0, 700000.0])  # 最佳参数
R = np.diag([0.2])
h_cm = 0.31

model = None
data = None
cam = None
scene = None
context = None
opt = None

aid_lw = -1
aid_rw = -1
aid_ls = -1
aid_rs = -1
adr_qpos_y = -1
adr_qvel_x = -1
adr_qvel_y = -1
adr_qpos_roll = -1
adr_qvel_roll = -1
adr_qpos_yaw = -1
adr_qvel_yaw = -1

adr_qpos_ls = -1
adr_qpos_rs = -1
adr_qvel_ls = -1
adr_qvel_rs = -1

wheel_radius = 0.09
x_front = 0.0
y_front = 0.0
x_rear = 0.0
y_rear = 0.0

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
prev_steer_f = 0.0
prev_steer_r = 0.0

# 数据记录变量
t_log = []
vx_cmd_log = []
vy_cmd_log = []
wz_cmd_log = []
vx_meas_log = []
vy_meas_log = []
wz_meas_log = []
y_ref_log = []
y_log = []
ydot_ref_log = []
ydot_log = []
roll_ref_log = []
roll_log = []
roll_dot_ref_log = []
roll_dot_log = []
u_y_log = []
u_ydot_log = []
u_roll_log = []
u_rolldot_log = []
fy_bal_log = []
fx_track_log = []
tau_yaw_log = []
steer_l_cmd_log = []
steer_r_cmd_log = []
steer_l_meas_log = []
steer_r_meas_log = []
tau_l_log = []
tau_r_log = []

# 舵角PID控制记录
steer_l_vel_ref_log = []  # 左舵角内环期望速度
steer_r_vel_ref_log = []  # 右舵角内环期望速度
steer_l_vel_meas_log = []  # 左舵角实际速度
steer_r_vel_meas_log = []  # 右舵角实际速度
steer_l_torque_log = []  # 左舵角输出扭矩
steer_r_torque_log = []  # 右舵角输出扭矩
steer_l_pos_err_log = []  # 左舵角位置误差
steer_r_pos_err_log = []  # 右舵角位置误差
steer_l_vel_err_log = []  # 左舵角速度误差
steer_r_vel_err_log = []  # 右舵角速度误差

class DualLoopPIDController:
    """双环PID控制器：外环角度控制，内环角速度控制"""
    def __init__(self, kp_pos, ki_pos, kd_pos, kp_vel, ki_vel, kd_vel, 
                 int_pos_min, int_pos_max, int_vel_min, int_vel_max,
                 out_min, out_max):
        # 外环（角度环）参数
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        self.int_pos_min = int_pos_min
        self.int_pos_max = int_pos_max
        self.int_pos = 0.0
        # 内环（角速度环）参数
        self.kp_vel = kp_vel
        self.ki_vel = ki_vel
        self.kd_vel = kd_vel
        self.int_vel_min = int_vel_min
        self.int_vel_max = int_vel_max
        self.int_vel = 0.0
        # 输出限幅
        self.out_min = out_min
        self.out_max = out_max
        # 上一次的角度误差（用于计算微分）
        self.last_pos_error = 0.0
    
    def compute(self, pos_ref, pos_meas, vel_meas, dt):
        """计算扭矩输出
        Args:
            pos_ref: 目标角度
            pos_meas: 当前角度
            vel_meas: 当前角速度
            dt: 时间步长
        Returns:
            (扭矩输出, 内环期望速度, 角度误差, 角速度误差)
        """
        # 外环：角度误差 -> 目标角速度
        pos_error = pos_ref - pos_meas
        self.int_pos = np.clip(self.int_pos + pos_error * dt, self.int_pos_min, self.int_pos_max)
        vel_ref = self.kp_pos * pos_error + self.ki_pos * self.int_pos + self.kd_pos * (pos_error - self.last_pos_error) / dt
        self.last_pos_error = pos_error
        
        # 内环：角速度误差 -> 扭矩
        vel_error = vel_ref - vel_meas
        self.int_vel = np.clip(self.int_vel + vel_error * dt, self.int_vel_min, self.int_vel_max)
        torque = self.kp_vel * vel_error + self.ki_vel * self.int_vel - self.kd_vel * vel_meas
        
        return np.clip(torque, self.out_min, self.out_max), vel_ref, pos_error, vel_error
    
    def reset(self):
        """重置积分项"""
        self.int_pos = 0.0
        self.int_vel = 0.0
        self.last_pos_error = 0.0


# 舵角双环PID控制器实例
steer_pid_left = DualLoopPIDController(
    kp_pos=50.0, ki_pos=5.0, kd_pos=2.0,  # 外环：角度PID
    kp_vel=5.0, ki_vel=0.5, kd_vel=0.2,   # 内环：角速度PID
    int_pos_min=-2.0, int_pos_max=2.0,
    int_vel_min=-5.0, int_vel_max=5.0,
    out_min=-15.0, out_max=15.0
)
steer_pid_right = DualLoopPIDController(
    kp_pos=50.0, ki_pos=5.0, kd_pos=2.0,
    kp_vel=5.0, ki_vel=0.5, kd_vel=0.2,
    int_pos_min=-2.0, int_pos_max=2.0,
    int_vel_min=-5.0, int_vel_max=5.0,
    out_min=-15.0, out_max=15.0
)

# 舵角关节索引
adr_qpos_steer_left = None
adr_qvel_steer_left = None
adr_qpos_steer_right = None
adr_qvel_steer_right = None

def get_joint_adr(joint_name):  # 获取关节索引
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # 关节 id
    qpos_adr = model.jnt_qposadr[jid]  # qpos 索引
    dof_adr = model.jnt_dofadr[jid]  # qvel 索引
    return qpos_adr, dof_adr  # 返回索引


def get_actuator_id(act_name):  # 获取执行器 id
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)  # 执行器 id


def clamp(v, vmin, vmax):
    return max(vmin, min(vmax, v))


def wrap_to_pi(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a

def angle_diff(a, b):
    return wrap_to_pi(a - b)


def get_state_from_data(data_in):
    y_wheel = data_in.qpos[adr_qpos_y]
    y_dot_wheel = data_in.qvel[adr_qvel_y]
    vx_world = float(data_in.qvel[adr_qvel_x])
    yaw = data_in.qpos[adr_qpos_yaw]
    phi = data_in.qpos[adr_qpos_roll]
    phi_dot = data_in.qvel[adr_qvel_roll]
    y_cm = y_wheel - h_cm * np.sin(phi)
    y_dot_cm = y_dot_wheel - h_cm * np.cos(phi) * phi_dot

    y_dot_cm = -vx_world * np.sin(yaw) + y_dot_cm * np.cos(yaw)

    return np.array([y_cm, y_dot_cm, phi, phi_dot], dtype=float)


def set_state_on_data(data_in, x_in, qpos_ref, qvel_ref):
    data_in.qpos[:] = qpos_ref
    data_in.qvel[:] = qvel_ref
    y_cm = x_in[0]
    y_dot_cm = x_in[1]
    phi = x_in[2]
    phi_dot = x_in[3]
    y_wheel = y_cm + h_cm * np.sin(phi)
    y_dot_wheel = y_dot_cm + h_cm * np.cos(phi) * phi_dot
    data_in.qpos[adr_qpos_y] = y_wheel
    data_in.qvel[adr_qvel_y] = y_dot_wheel
    data_in.qpos[adr_qpos_roll] = phi
    data_in.qvel[adr_qvel_roll] = phi_dot
    mj.mj_forward(model, data_in)


def g_discrete(x_in, u_in, qpos_ref, qvel_ref):
    data_lin = mj.MjData(model)
    set_state_on_data(data_lin, x_in, qpos_ref, qvel_ref)
    data_lin.ctrl[aid_lw] = 0.5 * u_in[0]
    data_lin.ctrl[aid_rw] = 0.5 * u_in[0]
    mj.mj_step(model, data_lin)
    return get_state_from_data(data_lin)


def linearize_discrete_central(x_eq, u_eq, qpos_ref, qvel_ref, eps=None):
    n = x_eq.shape[0]
    m = u_eq.shape[0]
    if eps is None:
        eps_x = np.full(n, 1e-5, dtype=float)
        eps_u = np.full(m, 1e-4, dtype=float)
    else:
        eps_x = eps[:n]
        eps_u = eps[n:] if len(eps) > n else np.full(m, 1e-3, dtype=float)
    Ad = np.zeros((n, n), dtype=float)
    Bd = np.zeros((n, m), dtype=float)
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps_x[i]
        x_plus = g_discrete(x_eq + dx, u_eq, qpos_ref, qvel_ref)
        x_minus = g_discrete(x_eq - dx, u_eq, qpos_ref, qvel_ref)
        Ad[:, i] = (x_plus - x_minus) / (2.0 * eps_x[i])
    for j in range(m):
        du = np.zeros(m, dtype=float)
        du[j] = eps_u[j]
        u_plus = g_discrete(x_eq, u_eq + du, qpos_ref, qvel_ref)
        u_minus = g_discrete(x_eq, u_eq - du, qpos_ref, qvel_ref)
        Bd[:, j] = (u_plus - u_minus) / (2.0 * eps_u[j])
        # print(Ad, Bd)
    return Ad, Bd


def init_LPV_controller():
    global K_interpolator, x0, qpos0, qvel0
    
    settle_time = 0.0
    while data.time < settle_time:
        data.ctrl[aid_lw] = 0.0
        data.ctrl[aid_rw] = 0.0
        mj.mj_step(model, data)
    
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    x0 = get_state_from_data(data)
    # x0[2] = 0.0
    
    print("开始进行 LPV 多点线性化与插值拟合...")
    roll_grid = np.linspace(-np.radians(60), np.radians(60), 31)
    K_list = []
    u0 = np.array([0.0], dtype=float)
    
    for roll in roll_grid:
        x_eq = x0.copy()
        x_eq[2] = roll
        x_eq[3] = 0.0
        Ad, Bd = linearize_discrete_central(x_eq, u0, qpos0, qvel0)
        try:
            P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
            K_i = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
            K_list.append(K_i[0])
        except Exception as e:
            print(f"在 roll = {np.degrees(roll):.2f}° 处求解失败: {e}")
            if len(K_list) > 0:
                K_list.append(K_list[-1])
            else:
                K_list.append(np.zeros(4))
    
    K_array = np.array(K_list)
    K_interpolator = CubicSpline(roll_grid, K_array)
    print("x0 =", x0)
    print("LPV 控制器初始化完成！")


def steer_solve(vx, vy, wz, x_i, y_i, current_steer=0.0):
    viy = vy + wz * x_i
    vix = vx - wz * y_i
    speed = math.hypot(vix, viy)
    
    if speed < 1e-6:
        return 0.0, 0.0
    
    target_steer = -math.atan2(vix, viy)
    
    err_direct = wrap_to_pi(target_steer - current_steer)
    err_reverse = wrap_to_pi(target_steer + math.pi - current_steer)
    
    if abs(err_reverse) < abs(err_direct):
        return wrap_to_pi(target_steer + math.pi), -speed / wheel_radius
    else:
        return target_steer, speed / wheel_radius

def steer_solve_combined(fy_bal, fx_track, tau_yaw, x_i, y_i, current_steer=0.0):
    force_X = fx_track
    force_Y = fy_bal + tau_yaw / x_i
    force = math.hypot(force_X, force_Y)

    if force < 1e-6:
        return 0.0, 0.0
    
    target_steer = -math.atan2(force_X, force_Y)

    err_direct = wrap_to_pi(target_steer - current_steer)
    err_reverse = wrap_to_pi(target_steer + math.pi - current_steer)
    
    if abs(err_reverse) < abs(err_direct):
        return wrap_to_pi(target_steer + math.pi), -force * wheel_radius
    else:
        return target_steer, force * wheel_radius


    return target_steer, force * wheel_radius
    


def keyboard(window, key, scancode, act, mods):
    global vx_cmd, vy_cmd, wz_cmd
    if act not in (glfw.PRESS, glfw.REPEAT):
        return
    if key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        return
    if key == glfw.KEY_UP:
        vy_cmd += v_step
    elif key == glfw.KEY_DOWN:
        vy_cmd -= v_step
    elif key == glfw.KEY_RIGHT:
        vx_cmd += v_step
    elif key == glfw.KEY_LEFT:
        vx_cmd -= v_step
    elif key == glfw.KEY_Q:
        wz_cmd += w_step
    elif key == glfw.KEY_E:
        wz_cmd -= w_step
    elif key == glfw.KEY_SPACE:
        vx_cmd = 0.0
        vy_cmd = 0.0
        wz_cmd = 0.0
    vx_cmd = clamp(vx_cmd, -v_max, v_max)
    vy_cmd = clamp(vy_cmd, -v_max, v_max)
    wz_cmd = clamp(wz_cmd, -w_max, w_max)


def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right, lastx, lasty
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    lastx, lasty = glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos
    if not (button_left or button_middle or button_right):
        return
    width, height = glfw.get_window_size(window)
    press_l = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    press_r = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = press_l or press_r
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


def init_ids():
    global aid_lw, aid_rw, aid_ls, aid_rs
    global adr_qpos_y, adr_qvel_x, adr_qvel_y, adr_qpos_roll, adr_qvel_roll, adr_qpos_yaw, adr_qvel_yaw
    global adr_qpos_ls, adr_qpos_rs, adr_qvel_ls, adr_qvel_rs
    global wheel_radius, x_front, y_front, x_rear, y_rear
    global adr_qpos_steer_left, adr_qvel_steer_left, adr_qpos_steer_right, adr_qvel_steer_right
    
    aid_lw = model.actuator("left_wheel_tau").id
    aid_rw = model.actuator("right_wheel_tau").id
    aid_ls = model.actuator("left_steer_tau").id
    aid_rs = model.actuator("right_steer_tau").id
    
    jid_y = model.joint("base_y").id
    jid_x = model.joint("base_x").id
    jid_roll = model.joint("base_roll").id
    jid_yaw = model.joint("base_yaw").id
    jid_ls = model.joint("left_steer").id
    jid_rs = model.joint("right_steer").id
    
    adr_qpos_y = model.jnt_qposadr[jid_y]
    adr_qvel_x = model.jnt_dofadr[jid_x]
    adr_qvel_y = model.jnt_dofadr[jid_y]
    adr_qpos_roll = model.jnt_qposadr[jid_roll]
    adr_qvel_roll = model.jnt_dofadr[jid_roll]
    adr_qpos_yaw = model.jnt_qposadr[jid_yaw]
    adr_qvel_yaw = model.jnt_dofadr[jid_yaw]
    adr_qpos_ls = model.jnt_qposadr[jid_ls]
    adr_qpos_rs = model.jnt_qposadr[jid_rs]
    adr_qvel_ls = model.jnt_dofadr[jid_ls]
    adr_qvel_rs = model.jnt_dofadr[jid_rs]

    adr_qpos_steer_left, adr_qvel_steer_left = get_joint_adr("left_steer")  # 左舵角索引
    adr_qpos_steer_right, adr_qvel_steer_right = get_joint_adr("right_steer")  # 右舵角索引
    
    wheel_radius = float(model.geom("left_wheel_geom").size[0])
    x_front = float(model.body("left_module").pos[0])
    y_front = float(model.body("left_module").pos[1])
    x_rear = float(model.body("right_module").pos[0])
    y_rear = float(model.body("right_module").pos[1])


def auto_test_command(t):
    """自动化测试：生成速度指令序列"""
    global vx_cmd, vy_cmd, wz_cmd
    if t < 2.0:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0
    elif t < 5.0:
        vx_cmd, vy_cmd, wz_cmd = 0.5, 0.0, 0.0
    elif t < 8.0:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.5, 0.0
    elif t < 11.0:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.5
    else:
        vx_cmd, vy_cmd, wz_cmd = 0.0, 0.0, 0.0


def controller(model_in, data_in):
    global prev_steer_f, prev_steer_r, vx_cmd, vy_cmd, wz_cmd
    global adr_qpos_yaw, adr_qvel_yaw
    dt = model_in.opt.timestep
    
    if K_interpolator is None:
        return
    
    if auto_test:
        auto_test_command(data_in.time)
    
    x_state = get_state_from_data(data_in)
    roll = x_state[2]
    
    K_current = K_interpolator(roll)
    y_ref = x0[0]
    ydot_ref = vy_cmd
    roll_ref = 0.0
    roll_dot_ref = 0.0
    x_ref = np.array([y_ref, ydot_ref, roll_ref, roll_dot_ref], dtype=float)
    e = x_state - x_ref
    u_bal = -K_current @ e
    fy_bal = clamp(float(u_bal), -5.0, 5.0) / max(wheel_radius, 1e-6) / 2
    
    vx_world = float(data_in.qvel[adr_qvel_x])
    vy_world = x_state[1]
    yaw_world = float(data_in.qpos[adr_qpos_yaw])
    vx_meas = vx_world * math.cos(yaw_world) + vy_world * math.sin(yaw_world)
    vy_meas = -vx_world * math.sin(yaw_world) + vy_world * math.cos(yaw_world)
    
    err_vx = vx_cmd - vx_meas
    fx_track = pid_vx.compute(err_vx, 0.0, dt)
    if abs(fx_track) < 3.0:
        fx_track = 0.0
    
    yaw_rate = float(data_in.qvel[adr_qvel_yaw])
    err_wz = wz_cmd - yaw_rate
    tau_yaw = pid_wz.compute(err_wz, 0.0, dt)
    
    speed_total = math.hypot(vx_cmd, vy_cmd) + abs(wz_cmd) * 0.5
    if speed_total < 1e-6:
        # pid_steer_l.reset()
        # pid_steer_r.reset()
        pid_vx.reset()
        pid_wz.reset()
    
    steer_l_pos = wrap_to_pi(data_in.qpos[adr_qpos_ls]) # 左舵角当前位置
    steer_r_pos = wrap_to_pi(data_in.qpos[adr_qpos_rs]) # 右舵角当前位置
    steer_l_vel = data_in.qvel[adr_qvel_steer_left]  # 左舵角当前角速度
    steer_r_vel = data_in.qvel[adr_qvel_steer_right]  # 右舵角当前角速度
    
    steer_l_des, torque_l = steer_solve_combined(fy_bal, fx_track, tau_yaw, x_front, y_front, steer_l_pos)
    steer_r_des, torque_r = steer_solve_combined(fy_bal, fx_track, tau_yaw, x_rear, y_rear, steer_r_pos)  

    tau_max = 5.0
        
    if not lock_steer:
        err_l = wrap_to_pi(steer_l_des - steer_l_pos)
        err_r = wrap_to_pi(steer_r_des - steer_r_pos)
        
        if err_l * err_r > 0 and abs(err_l) > 0.5 and abs(err_r) > 0.5:
            err_l_alt = wrap_to_pi(steer_l_des + math.pi - steer_l_pos)
            err_r_alt = wrap_to_pi(steer_r_des + math.pi - steer_r_pos)
            if abs(err_l_alt) + abs(err_r) < abs(err_l) + abs(err_r_alt):
                if abs(err_l_alt) < abs(err_l) + 0.5:
                    steer_l_des = wrap_to_pi(steer_l_des + math.pi)
                    torque_l = -torque_l
                    err_l = err_l_alt
            else:
                if abs(err_r_alt) < abs(err_r) + 0.5:
                    steer_r_des = wrap_to_pi(steer_r_des + math.pi)
                    torque_r = -torque_r
                    err_r = err_r_alt
                    
        if direct_steer_control:
            # 直接设置舵角位置（跳过执行器）
            data_in.qpos[adr_qpos_ls] = steer_l_pos + err_l
            data_in.qpos[adr_qpos_rs] = steer_r_pos + err_r
            data_in.qvel[adr_qvel_ls] = 0.0  # 舵角角速度设为0
            data_in.qvel[adr_qvel_rs] = 0.0
            data_in.ctrl[aid_ls] = 0.0  # 舵角扭矩设为0
            data_in.ctrl[aid_rs] = 0.0
            # data_in.ctrl[aid_lw] = float(clamp(torque_l, -tau_max, tau_max))
            # data_in.ctrl[aid_rw] = float(clamp(torque_r, -tau_max, tau_max))
            # 记录舵角PID数据（直接控制模式下）
            steer_l_vel_ref_log.append(0.0)
            steer_r_vel_ref_log.append(0.0)
            steer_l_vel_meas_log.append(steer_l_vel)
            steer_r_vel_meas_log.append(steer_r_vel)
            steer_l_torque_log.append(0.0)
            steer_r_torque_log.append(0.0)
            steer_l_pos_err_log.append(err_l)
            steer_r_pos_err_log.append(err_r)
            steer_l_vel_err_log.append(-steer_l_vel)
            steer_r_vel_err_log.append(-steer_r_vel)

        else: # 真实模式
            torque_l_pid, vel_ref_l, pos_err_l, vel_err_l = steer_pid_left.compute(err_l, 0, steer_l_vel, dt)
            torque_r_pid, vel_ref_r, pos_err_r, vel_err_r = steer_pid_right.compute(err_r, 0, steer_r_vel, dt)
            data_in.ctrl[aid_ls] = torque_l_pid  # 左舵角扭矩
            data_in.ctrl[aid_rs] = torque_r_pid  # 右舵角扭矩
            # 记录舵角PID数据
            steer_l_vel_ref_log.append(vel_ref_l)
            steer_r_vel_ref_log.append(vel_ref_r)
            steer_l_vel_meas_log.append(steer_l_vel)
            steer_r_vel_meas_log.append(steer_r_vel)
            steer_l_torque_log.append(torque_l_pid)
            steer_r_torque_log.append(torque_r_pid)
            steer_l_pos_err_log.append(pos_err_l)
            steer_r_pos_err_log.append(pos_err_r)
            steer_l_vel_err_log.append(vel_err_l)
            steer_r_vel_err_log.append(vel_err_r)

        if abs(err_l) < 0.1 and abs(err_r) < 0.1:
            data_in.ctrl[aid_lw] = float(clamp(torque_l, -tau_max, tau_max))
            data_in.ctrl[aid_rw] = float(clamp(torque_r, -tau_max, tau_max))
        else:
            data_in.ctrl[aid_lw] = 0.0
            data_in.ctrl[aid_rw] = 0.0
    
    else: # 舵角锁定在0度，测试， lock_steer = True
        # 舵角双环PID控制（目标角度为0，保持舵角归零）
        dt = model_in.opt.timestep
        steer_ref = 0.0  # 舵角目标为0度
        # steer_l_pos = data_in.qpos[adr_qpos_steer_left]  # 左舵角当前位置
        # steer_l_vel = data_in.qvel[adr_qvel_steer_left]  # 左舵角当前角速度
        # steer_r_pos = data_in.qpos[adr_qpos_steer_right]  # 右舵角当前位置
        # steer_r_vel = data_in.qvel[adr_qvel_steer_right]  # 右舵角当前角速度

        data_in.ctrl[aid_ls] = steer_pid_left.compute(steer_ref, steer_l_pos, steer_l_vel, dt)  # 左舵角扭矩
        data_in.ctrl[aid_rs] = steer_pid_right.compute(steer_ref, steer_r_pos, steer_r_vel, dt)  # 右舵角扭矩

        if abs(err_l) < 0.01 and abs(err_r) < 0.01:
            data_in.ctrl[aid_lw] = float(clamp(torque_l, -tau_max, tau_max))
            data_in.ctrl[aid_rw] = float(clamp(torque_r, -tau_max, tau_max))
        else:
            data_in.ctrl[aid_lw] = 0.0
            data_in.ctrl[aid_rw] = 0.0
        
    
    t_log.append(data_in.time)
    vx_cmd_log.append(vx_cmd)
    vy_cmd_log.append(vy_cmd)
    wz_cmd_log.append(wz_cmd)
    vx_meas_log.append(vx_meas)
    vy_meas_log.append(vy_meas)
    wz_meas_log.append(yaw_rate)
    y_ref_log.append(y_ref)
    y_log.append(x_state[0])
    ydot_ref_log.append(ydot_ref)
    ydot_log.append(x_state[1])
    roll_ref_log.append(roll_ref)
    roll_log.append(x_state[2])
    roll_dot_ref_log.append(roll_dot_ref)
    roll_dot_log.append(x_state[3])
    u_y_log.append(-K_current[0] * e[0])
    u_ydot_log.append(-K_current[1] * e[1])
    u_roll_log.append(-K_current[2] * e[2])
    u_rolldot_log.append(-K_current[3] * e[3])
    fy_bal_log.append(fy_bal)
    fx_track_log.append(fx_track)
    tau_yaw_log.append(tau_yaw / 0.13)
    steer_l_cmd_log.append(steer_l_des)
    steer_r_cmd_log.append(steer_r_des)
    steer_l_meas_log.append(steer_l_pos)
    steer_r_meas_log.append(steer_r_pos)
    tau_l_log.append(data_in.ctrl[aid_lw])
    tau_r_log.append(data_in.ctrl[aid_rw])


def main():
    global model, data, cam, scene, context, opt
    
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname, xml_path)
    model = mj.MjModel.from_xml_path(abspath)
    data = mj.MjData(model)
    init_ids()
    
    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    
    glfw.init()
    window = glfw.create_window(1200, 900, "LQR Balance + Swerve", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, scroll)
    
    mj.set_mjcb_control(controller)
    
    cam.azimuth = 90.0
    cam.elevation = -20.0
    cam.distance = 2.0
    
    init_LPV_controller()
    
    real_time_start = glfw.get_time()
    sim_time_start = data.time
    
    while not glfw.window_should_close(window):
        real_time_elapsed = glfw.get_time() - real_time_start
        target_sim_time = sim_time_start + real_time_elapsed * 1.0
        
        while data.time < target_sim_time:
            mj.mj_step(model, data)
            if data.time >= simend:
                break
        
        if data.time >= simend:
            glfw.set_window_should_close(window, True)
            break
        
        body_id = model.body("roll_base").id
        cam.lookat[:] = data.xpos[body_id]
        
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
    
    glfw.terminate()
    
    plot_results()


def plot_results():
    """绘制结果图像"""
    fig = plt.figure(figsize=(18, 14))
    
    # 1. vx速度命令与反馈
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.plot(t_log, vx_cmd_log, '--', label="vx_cmd")
    ax1.plot(t_log, vx_meas_log, label="vx_meas")
    ax1.set_ylabel("vx (m/s)")
    ax1.grid(True)
    ax1.legend()
    
    # 2. vy速度命令与反馈
    ax2 = fig.add_subplot(4, 4, 2)
    ax2.plot(t_log, vy_cmd_log, '--', label="vy_cmd")
    ax2.plot(t_log, vy_meas_log, label="vy_meas")
    ax2.set_ylabel("vy (m/s)")
    ax2.grid(True)
    ax2.legend()
    
    # 3. wz角速度命令与反馈
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.plot(t_log, wz_cmd_log, '--', label="wz_cmd")
    ax3.plot(t_log, wz_meas_log, label="wz_meas")
    ax3.set_ylabel("wz (rad/s)")
    ax3.grid(True)
    ax3.legend()
    
    # 4. y位置
    ax4 = fig.add_subplot(4, 4, 4)
    ax4.plot(t_log, y_log, label="y")
    ax4.plot(t_log, y_ref_log, '--', label="y_ref")
    ax4.set_ylabel("y (m)")
    ax4.grid(True)
    ax4.legend()
    
    # 5. y速度
    ax5 = fig.add_subplot(4, 4, 5)
    ax5.plot(t_log, ydot_log, label="ydot")
    ax5.plot(t_log, ydot_ref_log, '--', label="ydot_ref")
    ax5.set_ylabel("ydot (m/s)")
    ax5.grid(True)
    ax5.legend()
    
    # 6. roll角
    ax6 = fig.add_subplot(4, 4, 6)
    ax6.plot(t_log, np.degrees(roll_log), label="roll")
    ax6.plot(t_log, np.degrees(roll_ref_log), '--', label="roll_ref")
    ax6.set_ylabel("roll (deg)")
    ax6.grid(True)
    ax6.legend()
    
    # 7. roll角速度
    ax7 = fig.add_subplot(4, 4, 7)
    ax7.plot(t_log, roll_dot_log, label="roll_dot")
    ax7.plot(t_log, roll_dot_ref_log, '--', label="roll_dot_ref")
    ax7.set_ylabel("roll_dot (rad/s)")
    ax7.grid(True)
    ax7.legend()
    
    # 8. LQR四个力矩分量
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.plot(t_log, u_y_log, label="u_y")
    ax8.plot(t_log, u_ydot_log, label="u_ydot")
    ax8.plot(t_log, u_roll_log, label="u_roll")
    ax8.plot(t_log, u_rolldot_log, label="u_rolldot")
    ax8.set_ylabel("u components (Nm)")
    ax8.grid(True)
    ax8.legend()
    
    # 9. 三个力的大小对比
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.plot(t_log, fy_bal_log, label="fy_bal")
    ax9.plot(t_log, fx_track_log, label="fx_track")
    # ax9.plot(t_log, tau_yaw_log, label="tau_yaw/0.13")
    ax9.set_ylabel("force (N)")
    ax9.grid(True)
    ax9.legend()
    
    # 10. 舵角命令与实际（左舵）
    ax10 = fig.add_subplot(4, 4, 10)
    ax10.plot(t_log, np.degrees(steer_l_cmd_log), '--', label="steer_l_cmd")
    ax10.plot(t_log, np.degrees(steer_l_meas_log), label="steer_l_meas")
    ax10.set_ylabel("steer_l (deg)")
    ax10.grid(True)
    ax10.legend()
    
    # 11. 舵角命令与实际（右舵）
    ax11 = fig.add_subplot(4, 4, 11)
    # ax11.plot(t_log, np.degrees(steer_r_cmd_log), '--', label="steer_r_cmd")
    # ax11.plot(t_log, np.degrees(steer_r_meas_log), label="steer_r_meas")
    ax11.scatter(t_log, np.degrees(steer_r_cmd_log), s=1, marker='x', label="steer_r_cmd")
    ax11.scatter(t_log, np.degrees(steer_r_meas_log), s=1, marker='.', label="steer_r_meas")
    ax11.set_ylabel("steer_r (deg)")
    ax11.grid(True)
    ax11.legend()
    
    # 12. 轮子扭矩
    ax12 = fig.add_subplot(4, 4, 12)
    # ax12.plot(t_log, tau_l_log, label="tau_l")
    # ax12.plot(t_log, tau_r_log, label="tau_r")
    ax12.scatter(t_log, tau_l_log, s=1, marker='x', label="tau_l")
    ax12.scatter(t_log, tau_r_log, s=1, marker='.', label="tau_r")
    ax12.set_ylabel("wheel torque (Nm)")
    ax12.grid(True)
    ax12.legend()
    
    # 13. 舵角命令对比
    ax13 = fig.add_subplot(4, 4, 13)
    ax13.plot(t_log, np.degrees(steer_l_cmd_log), label="steer_l_cmd")
    ax13.plot(t_log, np.degrees(steer_r_cmd_log), label="steer_r_cmd")
    ax13.set_ylabel("steer cmd (deg)")
    ax13.set_xlabel("time (s)")
    ax13.grid(True)
    ax13.legend()
    
    # 14. 舵角实际对比
    ax14 = fig.add_subplot(4, 4, 14)
    ax14.plot(t_log, np.degrees(steer_l_meas_log), label="steer_l_meas")
    ax14.plot(t_log, np.degrees(steer_r_meas_log), label="steer_r_meas")
    ax14.set_ylabel("steer meas (deg)")
    ax14.set_xlabel("time (s)")
    ax14.grid(True)
    ax14.legend()
    
    # 15. 误差分析
    ax15 = fig.add_subplot(4, 4, 15)
    ax15.plot(t_log, np.array(y_log) - np.array(y_ref_log), label="y_err")
    ax15.plot(t_log, np.array(ydot_log) - np.array(ydot_ref_log), label="ydot_err")
    ax15.set_ylabel("error")
    ax15.set_xlabel("time (s)")
    ax15.grid(True)
    ax15.legend()
    
    # 16. roll误差
    ax16 = fig.add_subplot(4, 4, 16)
    ax16.plot(t_log, np.degrees(np.array(roll_log) - np.array(roll_ref_log)), label="roll_err")
    ax16.set_ylabel("roll error (deg)")
    ax16.set_xlabel("time (s)")
    ax16.grid(True)
    ax16.legend()
    
    plt.suptitle("Combined LQR Balance + Swerve Control Results", fontsize=14)
    plt.tight_layout()
    
    out_png = os.path.join(os.path.dirname(__file__), "combined_results.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"结果图像已保存到: {out_png}")
    
    # 输出统计信息
    print(f"roll角最大值: {np.max(np.abs(roll_log)):.4f} rad ({np.degrees(np.max(np.abs(roll_log))):.2f} deg)")
    print(f"roll角最终值: {roll_log[-1]:.4f} rad ({np.degrees(roll_log[-1]):.2f} deg)")
    print(f"y位置误差最大值: {np.max(np.abs(np.array(y_log) - np.array(y_ref_log))):.4f} m")
    
    # 绘制舵角PID控制数据
    plot_steer_pid_results()


def plot_steer_pid_results():
    """绘制舵角PID控制数据"""
    if len(steer_l_torque_log) == 0:
        print("没有舵角PID数据可绘制")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 左舵角期望vs实际角度
    ax1 = fig.add_subplot(3, 4, 1)
    ax1.plot(t_log, np.degrees(steer_l_cmd_log), '--', label="steer_l_des")
    ax1.plot(t_log, np.degrees(steer_l_meas_log), label="steer_l_meas")
    ax1.set_ylabel("angle (deg)")
    ax1.set_title("Left Steer Angle")
    ax1.grid(True)
    ax1.legend()
    
    # 2. 右舵角期望vs实际角度
    ax2 = fig.add_subplot(3, 4, 2)
    ax2.plot(t_log, np.degrees(steer_r_cmd_log), '--', label="steer_r_des")
    ax2.plot(t_log, np.degrees(steer_r_meas_log), label="steer_r_meas")
    ax2.set_ylabel("angle (deg)")
    ax2.set_title("Right Steer Angle")
    ax2.grid(True)
    ax2.legend()
    
    # 3. 左舵角位置误差
    ax3 = fig.add_subplot(3, 4, 3)
    ax3.plot(t_log, np.degrees(steer_l_pos_err_log), label="pos_err")
    ax3.set_ylabel("error (deg)")
    ax3.set_title("Left Steer Position Error")
    ax3.grid(True)
    ax3.legend()
    
    # 4. 右舵角位置误差
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.plot(t_log, np.degrees(steer_r_pos_err_log), label="pos_err")
    ax4.set_ylabel("error (deg)")
    ax4.set_title("Right Steer Position Error")
    ax4.grid(True)
    ax4.legend()
    
    # 5. 左舵角内环期望vs实际速度
    ax5 = fig.add_subplot(3, 4, 5)
    ax5.plot(t_log, np.degrees(steer_l_vel_ref_log), '--', label="vel_ref")
    ax5.plot(t_log, np.degrees(steer_l_vel_meas_log), label="vel_meas")
    ax5.set_ylabel("velocity (deg/s)")
    ax5.set_title("Left Steer Velocity")
    ax5.grid(True)
    ax5.legend()
    
    # 6. 右舵角内环期望vs实际速度
    ax6 = fig.add_subplot(3, 4, 6)
    ax6.plot(t_log, np.degrees(steer_r_vel_ref_log), '--', label="vel_ref")
    ax6.plot(t_log, np.degrees(steer_r_vel_meas_log), label="vel_meas")
    ax6.set_ylabel("velocity (deg/s)")
    ax6.set_title("Right Steer Velocity")
    ax6.grid(True)
    ax6.legend()
    
    # 7. 左舵角速度误差
    ax7 = fig.add_subplot(3, 4, 7)
    ax7.plot(t_log, np.degrees(steer_l_vel_err_log), label="vel_err")
    ax7.set_ylabel("error (deg/s)")
    ax7.set_title("Left Steer Velocity Error")
    ax7.grid(True)
    ax7.legend()
    
    # 8. 右舵角速度误差
    ax8 = fig.add_subplot(3, 4, 8)
    ax8.plot(t_log, np.degrees(steer_r_vel_err_log), label="vel_err")
    ax8.set_ylabel("error (deg/s)")
    ax8.set_title("Right Steer Velocity Error")
    ax8.grid(True)
    ax8.legend()
    
    # 9. 左舵角输出扭矩
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.plot(t_log, steer_l_torque_log, label="torque")
    ax9.set_ylabel("torque (Nm)")
    ax9.set_xlabel("time (s)")
    ax9.set_title("Left Steer Torque")
    ax9.grid(True)
    ax9.legend()
    
    # 10. 右舵角输出扭矩
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.plot(t_log, steer_r_torque_log, label="torque")
    ax10.set_ylabel("torque (Nm)")
    ax10.set_xlabel("time (s)")
    ax10.set_title("Right Steer Torque")
    ax10.grid(True)
    ax10.legend()
    
    # 11. 左右舵角扭矩对比
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.plot(t_log, steer_l_torque_log, label="left")
    ax11.plot(t_log, steer_r_torque_log, label="right")
    ax11.set_ylabel("torque (Nm)")
    ax11.set_xlabel("time (s)")
    ax11.set_title("Steer Torque Comparison")
    ax11.grid(True)
    ax11.legend()
    
    # 12. 左右舵角位置误差对比
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.plot(t_log, np.degrees(steer_l_pos_err_log), label="left")
    ax12.plot(t_log, np.degrees(steer_r_pos_err_log), label="right")
    ax12.set_ylabel("error (deg)")
    ax12.set_xlabel("time (s)")
    ax12.set_title("Position Error Comparison")
    ax12.grid(True)
    ax12.legend()
    
    plt.suptitle("Steer PID Control Details", fontsize=14)
    plt.tight_layout()
    
    out_png = os.path.join(os.path.dirname(__file__), "steer_pid_results.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"舵角PID控制图像已保存到: {out_png}")
    print(f"y速度误差最大值: {np.max(np.abs(np.array(ydot_log) - np.array(ydot_ref_log))):.4f} m/s")
    print(f"vx速度误差最大值: {np.max(np.abs(np.array(vx_meas_log) - np.array(vx_cmd_log))):.4f} m/s")
    print(f"轮子扭矩最大值: {np.max(np.abs(tau_l_log)):.4f} Nm")


if __name__ == "__main__":
    main()
