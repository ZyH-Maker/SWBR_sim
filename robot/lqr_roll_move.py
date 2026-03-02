import os  # 系统路径
import math  # 数学函数
import numpy as np  # 数值计算
import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # GLFW 绑定

xml_path = "scene.xml"  # 场景文件
sim_end = 60.0  # 仿真时长
settle_time = 0.1  # 初始稳定时间

t_hold = 20.0  # 0-10s 只做平衡
t_move = 60.0  # 10-20s 加入前进速度
vx_move = 0.1  # 前进速度目标

K = np.array([[76.73618566, 8.13251395]])  # LQR 增益占位（填入 roll2d 结果）

kv_x = 50.0  # 前进速度控制比例
fx_max = 200.0  # 前进力限幅
fy_max = 200.0  # 侧向力限幅

model = None  # MuJoCo 模型
data = None  # MuJoCo 数据
cam = None  # 相机
scene = None  # 场景
context = None  # 渲染上下文
opt = None  # 可视化选项

aid_ls = -1  # 左舵角执行器 id
aid_rs = -1  # 右舵角执行器 id
aid_lw = -1  # 左轮力矩执行器 id
aid_rw = -1  # 右轮力矩执行器 id

adr_qpos_roll = -1  # roll 关节 qpos 索引
adr_qvel_roll = -1  # roll 关节 qvel 索引
adr_qvel_x = -1  # base_x 关节 qvel 索引

wheel_radius = 0.05  # 轮子半径
steer_cmd = 0.0  # 当前舵角命令
t_start = 0.0  # 控制开始时间

button_left = False  # 鼠标左键
button_middle = False  # 鼠标中键
button_right = False  # 鼠标右键
lastx = 0  # 鼠标 x
lasty = 0  # 鼠标 y


def clamp(v, vmin, vmax):  # 数值限幅
    return max(vmin, min(vmax, v))  # 返回限幅结果


def get_state():  # 读取 roll 状态
    roll = float(data.qpos[adr_qpos_roll])  # roll 角
    roll_dot = float(data.qvel[adr_qvel_roll])  # roll 角速度
    return np.array([roll, roll_dot], dtype=float)  # 状态向量


def get_vx():  # 读取前进速度
    return float(data.qvel[adr_qvel_x])  # base_x 速度


def keyboard(window, key, scancode, act, mods):  # 键盘回调
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:  # 退格重置
        mj.mj_resetData(model, data)  # 重置数据
        mj.mj_forward(model, data)  # 前向更新


def mouse_button(window, button, act, mods):  # 鼠标按键回调
    global button_left, button_middle, button_right, lastx, lasty  # 使用全局变量
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)  # 左键
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)  # 中键
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)  # 右键
    lastx, lasty = glfw.get_cursor_pos(window)  # 记录鼠标位置


def mouse_move(window, xpos, ypos):  # 鼠标移动回调
    global lastx, lasty, button_left, button_middle, button_right  # 使用全局变量
    dx = xpos - lastx  # x 位移
    dy = ypos - lasty  # y 位移
    lastx = xpos  # 更新 x
    lasty = ypos  # 更新 y
    if not (button_left or button_middle or button_right):  # 无按键
        return  # 直接返回
    width, height = glfw.get_window_size(window)  # 窗口尺寸
    press_l = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS  # 左 shift
    press_r = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS  # 右 shift
    mod_shift = press_l or press_r  # 是否按下 shift
    if button_right:  # 右键
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V  # 平移
    elif button_left:  # 左键
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V  # 旋转
    else:  # 中键
        action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放
    if cam is not None and scene is not None and model is not None:  # 检查对象
        mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # 更新相机


def scroll(window, xoffset, yoffset):  # 滚轮回调
    if cam is not None and scene is not None and model is not None:  # 检查对象
        mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)  # 更新相机


def init_ids():  # 初始化索引
    global aid_ls, aid_rs, aid_lw, aid_rw  # 执行器 id
    global adr_qpos_roll, adr_qvel_roll, adr_qvel_x  # 关节索引
    global wheel_radius  # 轮子半径
    aid_ls = model.actuator("left_steer_pos").id  # 左舵角执行器
    aid_rs = model.actuator("right_steer_pos").id  # 右舵角执行器
    aid_lw = model.actuator("left_wheel_tau").id  # 左轮力矩执行器
    aid_rw = model.actuator("right_wheel_tau").id  # 右轮力矩执行器
    jid_roll = model.joint("base_roll").id  # roll 关节 id
    jid_x = model.joint("base_x").id  # base_x 关节 id
    adr_qpos_roll = model.jnt_qposadr[jid_roll]  # roll qpos 索引
    adr_qvel_roll = model.jnt_dofadr[jid_roll]  # roll qvel 索引
    adr_qvel_x = model.jnt_dofadr[jid_x]  # base_x qvel 索引
    wheel_radius = float(model.geom("left_wheel_geom").size[0])  # 轮子半径


def controller(model_in, data_in):  # 控制回调
    global steer_cmd  # 使用舵角命令
    t = data_in.time - t_start  # 控制相对时间
    x = get_state()  # 当前 roll 状态
    e = x - np.array([0.0, 0.0], dtype=float)  # 只平衡 roll
    u = -K @ e  # LQR 控制量（总力矩）
    u = float(u[0])  # 转成标量
    fy_cmd = u / max(wheel_radius, 1e-6)  # 侧向力目标
    fy_cmd = clamp(fy_cmd, -fy_max, fy_max)  # 侧向力限幅
    if t < t_hold:  # 0-10s 固定侧向滚动
        steer_target = math.pi / 2.0  # 舵角固定 90 度
        tau_each = 0.5 * u  # 直接按 u/2 输出力矩
    elif t < t_move:  # 10-20s 进入前进阶段
        vx_err = vx_move - get_vx()  # 前进速度误差
        fx_cmd = clamp(kv_x * vx_err, -fx_max, fx_max)  # 前进力命令
        f_vec_x = fx_cmd  # 合力 x 分量（来自前进控制）
        f_vec_y = fy_cmd  # 合力 y 分量（来自 LQR 的 u）
        steer_target = math.atan2(f_vec_y, f_vec_x)  # 舵角方向
        f_mag = math.hypot(f_vec_x, f_vec_y)  # 合力大小
        tau_each = 0.5 * f_mag * wheel_radius  # 每轮力矩
    steer_cmd = steer_target  # 更新舵角命令
    data_in.ctrl[aid_ls] = float(steer_target)  # 左舵角目标
    data_in.ctrl[aid_rs] = float(steer_target)  # 右舵角目标
    data_in.ctrl[aid_lw] = float(tau_each)  # 左轮力矩
    data_in.ctrl[aid_rw] = float(tau_each)  # 右轮力矩


def main():  # 主函数
    global model, data, cam, scene, context, opt, t_start  # 全局对象
    dirname = os.path.dirname(__file__)  # 当前目录
    abspath = os.path.join(dirname, xml_path)  # 模型路径
    model = mj.MjModel.from_xml_path(abspath)  # 加载模型
    data = mj.MjData(model)  # 创建数据
    init_ids()  # 初始化索引
    cam = mj.MjvCamera()  # 相机
    opt = mj.MjvOption()  # 选项
    mj.mjv_defaultCamera(cam)  # 默认相机
    mj.mjv_defaultOption(opt)  # 默认选项
    glfw.init()  # 初始化 GLFW
    window = glfw.create_window(1200, 900, "Roll Balance + Move", None, None)  # 创建窗口
    glfw.make_context_current(window)  # 绑定上下文
    glfw.swap_interval(1)  # 垂直同步
    scene = mj.MjvScene(model, maxgeom=20000)  # 场景
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染
    glfw.set_key_callback(window, keyboard)  # 键盘回调
    glfw.set_cursor_pos_callback(window, mouse_move)  # 鼠标移动回调
    glfw.set_mouse_button_callback(window, mouse_button)  # 鼠标按键回调
    glfw.set_scroll_callback(window, scroll)  # 滚轮回调
    while data.time < settle_time:  # 初始稳定阶段
        mj.mj_step(model, data)  # 仿真步进
    t_start = data.time  # 记录控制起点
    mj.set_mjcb_control(controller)  # 注册控制器
    cam.azimuth = 90.0  # 相机方位角
    cam.elevation = -20.0  # 相机仰角
    cam.distance = 2.0  # 相机距离
    t0 = data.time  # 起始时间
    while not glfw.window_should_close(window) and data.time - t0 < sim_end:  # 主循环
        time_prev = data.time  # 上一时刻
        while data.time - time_prev < 1.0 / 60.0:  # 60Hz 渲染
            mj.mj_step(model, data)  # 物理步进
        viewport_w, viewport_h = glfw.get_framebuffer_size(window)  # 视口
        viewport = mj.MjrRect(0, 0, viewport_w, viewport_h)  # 视口矩形
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)  # 更新场景
        mj.mjr_render(viewport, scene, context)  # 渲染
        glfw.swap_buffers(window)  # 交换缓冲
        glfw.poll_events()  # 处理事件
    glfw.terminate()  # 释放资源


if __name__ == "__main__":  # 入口判断
    main()  # 运行主程序
