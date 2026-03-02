import os  # 系统路径相关
import math  # 数学函数
import numpy as np  # 数值计算
import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # MuJoCo 的 GLFW

xml_path = "steer_cart.xml"  # 模型文件名
sim_end = 3000.0  # 仿真总时间

vx_cmd = 0.0  # 期望 x 方向速度
vy_cmd = 0.0  # 期望 y 方向速度
wz_cmd = 0.0  # 期望 yaw 角速度

v_step = 0.1  # 速度增量
w_step = 0.2  # 角速度增量
v_max = 2.0  # 速度上限
w_max = 2.0  # 角速度上限

kp_steer = 40.0  # 舵角 PID 的比例
kd_steer = 2.0  # 舵角 PID 的微分
ki_steer = 0.5  # 舵角 PID 的积分
kp_wheel = 10.0  # 轮速 PID 的比例
kd_wheel = 0.5  # 轮速 PID 的微分
ki_wheel = 0.5  # 轮速 PID 的积分

tau_steer_max = 50.0  # 舵角力矩限幅
tau_wheel_max = 50.0  # 轮子力矩限幅

int_steer_l = 0.0  # 左舵角积分
int_steer_r = 0.0  # 右舵角积分
int_wheel_l = 0.0  # 左轮积分
int_wheel_r = 0.0  # 右轮积分

model = None  # MuJoCo 模型句柄
data = None  # MuJoCo 数据句柄
cam = None  # 相机
scene = None  # 场景
context = None  # 渲染上下文

aid_left_steer = -1  # 左舵角执行器 id
aid_right_steer = -1  # 右舵角执行器 id
aid_left_wheel = -1  # 左轮执行器 id
aid_right_wheel = -1  # 右轮执行器 id

qpos_left_steer = -1  # 左舵角 qpos 索引
qvel_left_steer = -1  # 左舵角 qvel 索引
qpos_right_steer = -1  # 右舵角 qpos 索引
qvel_right_steer = -1  # 右舵角 qvel 索引
qvel_left_wheel = -1  # 左轮 qvel 索引
qvel_right_wheel = -1  # 右轮 qvel 索引

wheel_radius = 0.05  # 轮子半径
x_front = 0.13  # 前轮模块 x 坐标
y_front = 0.0  # 前轮模块 y 坐标
x_rear = -0.13  # 后轮模块 x 坐标
y_rear = 0.0  # 后轮模块 y 坐标

chassis_bid = -1  # 车身 body id


def clamp(v, vmin, vmax):  # 截断函数
    return max(vmin, min(vmax, v))  # 返回截断值


def wrap_to_pi(a):  # 角度归一化到 [-pi, pi]
    while a > math.pi:  # 大于 pi 继续减
        a -= 2.0 * math.pi  # 角度减 2pi
    while a < -math.pi:  # 小于 -pi 继续加
        a += 2.0 * math.pi  # 角度加 2pi
    return a  # 返回归一化角度


def steer_solve(vx, vy, wz, x_i, y_i):  # 舵轮解算
    vix = vx - wz * y_i  # 模块处 x 速度
    viy = vy + wz * x_i  # 模块处 y 速度
    steer = math.atan2(viy, vix)  # 舵角
    speed = math.hypot(vix, viy)  # 线速度
    wheel_omega = speed / max(1e-6, wheel_radius)  # 轮速
    return steer, wheel_omega  # 返回舵角与轮速


def keyboard(window, key, scancode, act, mods):  # 键盘回调
    global vx_cmd, vy_cmd, wz_cmd  # 使用全局命令
    if act not in (glfw.PRESS, glfw.REPEAT):  # 只处理按下/长按
        return  # 直接返回
    if key == glfw.KEY_UP:  # 前进
        vx_cmd += v_step  # 增加 x 速度
    elif key == glfw.KEY_DOWN:  # 后退
        vx_cmd -= v_step  # 减少 x 速度
    elif key == glfw.KEY_RIGHT:  # 向右平移
        vy_cmd -= v_step  # 减少 y 速度
    elif key == glfw.KEY_LEFT:  # 向左平移
        vy_cmd += v_step  # 增加 y 速度
    elif key == glfw.KEY_Q:  # 逆时针旋转
        wz_cmd += w_step  # 增加 yaw 速度
    elif key == glfw.KEY_E:  # 顺时针旋转
        wz_cmd -= w_step  # 减少 yaw 速度
    elif key == glfw.KEY_SPACE:  # 清零
        vx_cmd = 0.0  # 清零 x
        vy_cmd = 0.0  # 清零 y
        wz_cmd = 0.0  # 清零 yaw
    vx_cmd = clamp(vx_cmd, -v_max, v_max)  # 速度限幅
    vy_cmd = clamp(vy_cmd, -v_max, v_max)  # 速度限幅
    wz_cmd = clamp(wz_cmd, -w_max, w_max)  # 角速度限幅


def init_ids():  # 初始化索引与几何
    global aid_left_steer, aid_right_steer, aid_left_wheel, aid_right_wheel  # 执行器 id
    global qpos_left_steer, qvel_left_steer, qpos_right_steer, qvel_right_steer  # 舵角索引
    global qvel_left_wheel, qvel_right_wheel  # 轮子索引
    global wheel_radius, x_front, y_front, x_rear, y_rear, chassis_bid  # 几何与坐标
    aid_left_steer = model.actuator("left_steer_tau").id  # 左舵角执行器
    aid_right_steer = model.actuator("right_steer_tau").id  # 右舵角执行器
    aid_left_wheel = model.actuator("left_wheel_tau").id  # 左轮执行器
    aid_right_wheel = model.actuator("right_wheel_tau").id  # 右轮执行器
    jid_ls = model.joint("left_steer").id  # 左舵角关节
    jid_rs = model.joint("right_steer").id  # 右舵角关节
    jid_lw = model.joint("left_wheel_spin").id  # 左轮关节
    jid_rw = model.joint("right_wheel_spin").id  # 右轮关节
    qpos_left_steer = model.jnt_qposadr[jid_ls]  # 左舵角 qpos
    qvel_left_steer = model.jnt_dofadr[jid_ls]  # 左舵角 qvel
    qpos_right_steer = model.jnt_qposadr[jid_rs]  # 右舵角 qpos
    qvel_right_steer = model.jnt_dofadr[jid_rs]  # 右舵角 qvel
    qvel_left_wheel = model.jnt_dofadr[jid_lw]  # 左轮 qvel
    qvel_right_wheel = model.jnt_dofadr[jid_rw]  # 右轮 qvel
    wheel_radius = model.geom("left_wheel_geom").size[0]  # 轮子半径
    x_front, y_front, _ = model.body("left_module").pos  # 前轮模块坐标
    x_rear, y_rear, _ = model.body("right_module").pos  # 后轮模块坐标
    chassis_bid = model.body("chassis").id  # 车身 body id


def controller(model_in, data_in):  # 控制回调
    global int_steer_l, int_steer_r, int_wheel_l, int_wheel_r  # 积分项
    dt = model_in.opt.timestep  # 时间步长
    steer_f, omega_f = steer_solve(vx_cmd, vy_cmd, wz_cmd, x_front, y_front)  # 前轮目标
    steer_r, omega_r = steer_solve(vx_cmd, vy_cmd, wz_cmd, x_rear, y_rear)  # 后轮目标
    theta_l = data_in.qpos[qpos_left_steer]  # 左舵角
    theta_r = data_in.qpos[qpos_right_steer]  # 右舵角
    dtheta_l = data_in.qvel[qvel_left_steer]  # 左舵角角速度
    dtheta_r = data_in.qvel[qvel_right_steer]  # 右舵角角速度
    w_l = data_in.qvel[qvel_left_wheel]  # 左轮角速度
    w_r = data_in.qvel[qvel_right_wheel]  # 右轮角速度
    err_ls = wrap_to_pi(steer_f - theta_l)  # 左舵角误差
    err_rs = wrap_to_pi(steer_r - theta_r)  # 右舵角误差
    int_steer_l = clamp(int_steer_l + err_ls * dt, -1.0, 1.0)  # 左舵角积分
    int_steer_r = clamp(int_steer_r + err_rs * dt, -1.0, 1.0)  # 右舵角积分
    tau_ls = kp_steer * err_ls - kd_steer * dtheta_l + ki_steer * int_steer_l  # 左舵角力矩
    tau_rs = kp_steer * err_rs - kd_steer * dtheta_r + ki_steer * int_steer_r  # 右舵角力矩
    err_wl = omega_f - w_l  # 左轮速度误差
    err_wr = omega_r - w_r  # 右轮速度误差
    int_wheel_l = clamp(int_wheel_l + err_wl * dt, -2.0, 2.0)  # 左轮积分
    int_wheel_r = clamp(int_wheel_r + err_wr * dt, -2.0, 2.0)  # 右轮积分
    tau_wl = kp_wheel * err_wl - kd_wheel * w_l + ki_wheel * int_wheel_l  # 左轮力矩
    tau_wr = kp_wheel * err_wr - kd_wheel * w_r + ki_wheel * int_wheel_r  # 右轮力矩
    data_in.ctrl[aid_left_steer] = float(clamp(tau_ls, -tau_steer_max, tau_steer_max))  # 左舵角输出
    data_in.ctrl[aid_right_steer] = float(clamp(tau_rs, -tau_steer_max, tau_steer_max))  # 右舵角输出
    data_in.ctrl[aid_left_wheel] = float(clamp(tau_wl, -tau_wheel_max, tau_wheel_max))  # 左轮输出
    data_in.ctrl[aid_right_wheel] = float(clamp(tau_wr, -tau_wheel_max, tau_wheel_max))  # 右轮输出


def main():  # 主函数
    global model, data, cam, scene, context  # 共享对象
    dirname = os.path.dirname(__file__)  # 脚本目录
    abspath = os.path.join(dirname, xml_path)  # 模型路径
    model = mj.MjModel.from_xml_path(abspath)  # 加载模型
    data = mj.MjData(model)  # 创建数据
    init_ids()  # 初始化索引
    cam = mj.MjvCamera()  # 创建相机
    opt = mj.MjvOption()  # 可视化选项
    mj.mjv_defaultCamera(cam)  # 相机默认
    mj.mjv_defaultOption(opt)  # 选项默认
    glfw.init()  # 初始化 GLFW
    window = glfw.create_window(1200, 900, "steer cart", None, None)  # 创建窗口
    glfw.make_context_current(window)  # 绑定上下文
    glfw.swap_interval(1)  # 垂直同步
    scene = mj.MjvScene(model, maxgeom=20000)  # 场景
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染上下文
    glfw.set_key_callback(window, keyboard)  # 注册键盘回调
    mj.set_mjcb_control(controller)  # 注册控制回调
    cam.azimuth = 90.0  # 相机方位角
    cam.elevation = -20.0  # 相机仰角
    cam.distance = 2.0  # 相机距离
    t0 = data.time  # 起始时间
    while not glfw.window_should_close(window):  # 主循环
        time_prev = data.time  # 上一次时间
        while data.time - time_prev < 1.0 / 60.0:  # 控制显示频率
            mj.mj_step(model, data)  # 物理步进
        if data.time - t0 > sim_end:  # 结束条件
            break  # 结束仿真
        if chassis_bid >= 0:  # 如果车身存在
            cam.lookat[:] = data.xpos[chassis_bid]  # 相机跟随车身
        viewport_w, viewport_h = glfw.get_framebuffer_size(window)  # 视口大小
        viewport = mj.MjrRect(0, 0, viewport_w, viewport_h)  # 视口矩形
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)  # 更新场景
        mj.mjr_render(viewport, scene, context)  # 渲染
        glfw.swap_buffers(window)  # 交换缓冲
        glfw.poll_events()  # 处理事件
    glfw.terminate()  # 释放资源


if __name__ == "__main__":  # 入口判断
    main()  # 运行主程序
