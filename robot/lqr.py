import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # MuJoCo 的 GLFW 绑定
import numpy as np  # 数值计算
import os  # 路径处理
import control  # LQR 工具库


# ======================== 基本参数 ======================== 
xml_path = "ip2d.xml"  # 模型文件（与本脚本同目录）
simend = 100.0  # 仿真结束时间（秒）
settle_time = 0.1  # 先空跑让其落地稳定（秒）
print_camera_config = 0  # 是否打印相机参数


# ======================== 全局变量 ========================
model = None  # MuJoCo 模型
data = None  # MuJoCo 数据
cam = None  # 相机
scene = None  # 场景
context = None  # 渲染上下文

K = None  # LQR 增益矩阵
x0 = None  # 线性化工作点状态
qpos0 = None  # 工作点 qpos
qvel0 = None  # 工作点 qvel

# 关节与执行器索引
jid_base_y = None  # y 平移关节 id
jid_base_roll = None  # roll 关节 id
adr_qpos_y = None  # y 在 qpos 中的索引
adr_qvel_y = None  # y 在 qvel 中的索引
adr_qpos_roll = None  # roll 在 qpos 中的索引
adr_qvel_roll = None  # roll 在 qvel 中的索引

aid_left = None  # 左轮力矩执行器 id
aid_right = None  # 右轮力矩执行器 id


# ======================== 工具函数 ======================== 
def get_joint_adr(joint_name):
    """根据关节名获取 qpos/qvel 索引（中文注释）"""
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # 关节 id
    qpos_adr = model.jnt_qposadr[jid]  # qpos 索引
    dof_adr = model.jnt_dofadr[jid]  # qvel 索引
    return jid, qpos_adr, dof_adr  # 返回索引


def get_actuator_id(act_name):
    """根据执行器名获取 id（中文注释）"""
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)  # 执行器 id


def get_state_from_data(data_in):
    """从 MuJoCo 读取二维 roll 状态 x = [y, y_dot, phi, phi_dot]（中文注释）"""
    y = data_in.qpos[adr_qpos_y]  # y 位置
    y_dot = data_in.qvel[adr_qvel_y]  # y 速度
    phi = data_in.qpos[adr_qpos_roll]  # roll 角
    phi_dot = data_in.qvel[adr_qvel_roll]  # roll 角速度
    return np.array([y, y_dot, phi, phi_dot], dtype=float)  # 返回状态向量


def set_state_on_data(data_in, x_in, qpos_ref, qvel_ref):
    """把状态写回 MuJoCo（只改 y 与 roll，其它保持参考值）（中文注释）"""
    data_in.qpos[:] = qpos_ref  # 先还原参考 qpos
    data_in.qvel[:] = qvel_ref  # 先还原参考 qvel
    data_in.qpos[adr_qpos_y] = x_in[0]  # 写 y
    data_in.qvel[adr_qvel_y] = x_in[1]  # 写 y_dot
    data_in.qpos[adr_qpos_roll] = x_in[2]  # 写 roll
    data_in.qvel[adr_qvel_roll] = x_in[3]  # 写 roll_dot
    mj.mj_forward(model, data_in)  # 前向计算更新


def g_discrete(x_in, u_in, qpos_ref, qvel_ref):
    """离散一步映射 x_next = g(x,u)（中文注释）"""
    data_lin = mj.MjData(model)  # 使用独立数据避免污染主仿真
    set_state_on_data(data_lin, x_in, qpos_ref, qvel_ref)  # 设置状态
    data_lin.ctrl[aid_left] = 0.5 * u_in[0]  # 左轮力矩
    data_lin.ctrl[aid_right] = 0.5 * u_in[0]  # 右轮力矩
    mj.mj_step(model, data_lin)  # 前进一步
    return get_state_from_data(data_lin)  # 返回下一步状态


def linearize_discrete(x_eq, u_eq, qpos_ref, qvel_ref, eps=None):
    """数值线性化离散系统（中文注释）"""
    if eps is None:
        eps = np.array([1e-4, 1e-3, 1e-4, 1e-3], dtype=float)  # 扰动幅值
    n = x_eq.shape[0]  # 状态维数
    m = u_eq.shape[0]  # 输入维数
    Ad = np.zeros((n, n), dtype=float)  # 离散 A
    Bd = np.zeros((n, m), dtype=float)  # 离散 B
    x0_next = g_discrete(x_eq, u_eq, qpos_ref, qvel_ref)  # 基准步进
    for i in range(n):  # 对状态逐维扰动
        dx = np.zeros(n, dtype=float)  # 状态扰动
        dx[i] = eps[i]  # 设置扰动
        xi_next = g_discrete(x_eq + dx, u_eq, qpos_ref, qvel_ref)  # 扰动步进
        Ad[:, i] = (xi_next - x0_next) / eps[i]  # 差分近似
    for j in range(m):  # 对输入逐维扰动
        du = np.zeros(m, dtype=float)  # 输入扰动
        du[j] = 1e-2  # 输入扰动幅值（力矩）
        ui_next = g_discrete(x_eq, u_eq + du, qpos_ref, qvel_ref)  # 扰动步进
        Bd[:, j] = (ui_next - x0_next) / du[j]  # 差分近似
    return Ad, Bd  # 返回离散 A、B


# ======================== 控制器初始化 ======================== 
def init_controller():
    """初始化 LQR（中文注释）"""
    global K, x0, qpos0, qvel0  # 使用全局变量

    # 先空跑让其落地稳定（中文注释）
    while data.time < settle_time:
        data.ctrl[aid_left] = 0.0  # 左轮无力矩
        data.ctrl[aid_right] = 0.0  # 右轮无力矩
        mj.mj_step(model, data)  # 仿真一步

    # 记录工作点（中文注释）
    qpos0 = data.qpos.copy()  # 记录 qpos
    qvel0 = data.qvel.copy()  # 记录 qvel
    x0 = get_state_from_data(data)  # 记录平衡点状态
    u0 = np.array([0.0], dtype=float)  # 平衡输入

    # 数值线性化（中文注释）
    Ad, Bd = linearize_discrete(x0, u0, qpos0, qvel0)  # 离散线性化

    # LQR 权重（中文注释）
    Q = np.diag([200.0, 20.0, 500.0, 50.0])  # 状态权重
    R = np.diag([1.0])  # 输入权重

    # 离散 LQR（中文注释）
    K, S, E = control.dlqr(Ad, Bd, Q, R)  # 计算 K

    # 打印调试信息（中文注释）
    print("x0 =", x0)  # 输出工作点
    print("LQR K =", K)  # 输出增益矩阵


def controller(model, data):
    """MuJoCo 控制回调（中文注释）"""
    x = get_state_from_data(data)  # 读取当前状态
    e = x - x0  # 状态误差
    u = -K @ e  # LQR 控制
    # 力矩限幅（中文注释）
    u = np.clip(u, -200.0, 200.0)  # 根据执行器范围限幅
    # 两轮平均分配（中文注释）
    data.ctrl[aid_left] = 0.5 * u[0]  # 左轮力矩
    data.ctrl[aid_right] = 0.5 * u[0]  # 右轮力矩


# ======================== GLFW 回调函数 ======================== 
button_left = False  # 鼠标左键
button_middle = False  # 鼠标中键
button_right = False  # 鼠标右键
lastx = 0  # 鼠标 x
lasty = 0  # 鼠标 y


def keyboard(window, key, scancode, act, mods):
    """键盘事件：按下 Backspace 重置仿真（中文注释）"""
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)  # 重置数据
        mj.mj_forward(model, data)  # 重新前向计算


def mouse_button(window, button, act, mods):
    """鼠标按键事件（中文注释）"""
    global button_left, button_middle, button_right  # 使用全局变量
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)  # 左键
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)  # 中键
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)  # 右键
    glfw.get_cursor_pos(window)  # 更新鼠标位置


def mouse_move(window, xpos, ypos):
    """鼠标移动事件（中文注释）"""
    global lastx, lasty, button_left, button_middle, button_right  # 使用全局变量
    dx = xpos - lastx  # x 位移
    dy = ypos - lasty  # y 位移
    lastx = xpos  # 更新 x
    lasty = ypos  # 更新 y
    if not (button_left or button_middle or button_right):  # 无按键则退出
        return
    width, height = glfw.get_window_size(window)  # 窗口尺寸
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS  # Shift
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS  # Shift
    mod_shift = PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT  # 是否按下 Shift
    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V  # 平移相机
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V  # 旋转相机
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放相机
    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # 更新相机


def scroll(window, xoffset, yoffset):
    """鼠标滚轮事件（中文注释）"""
    action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)  # 更新相机


# ======================== 主程序 ======================== 
def main():
    global model, data, cam, scene, context  # 使用全局变量
    global jid_base_y, jid_base_roll, adr_qpos_y, adr_qvel_y, adr_qpos_roll, adr_qvel_roll  # 关节索引
    global aid_left, aid_right  # 执行器索引

    # 加载模型（中文注释）
    dirname = os.path.dirname(__file__)  # 当前目录
    abspath = os.path.join(dirname, xml_path)  # 模型绝对路径
    model = mj.MjModel.from_xml_path(abspath)  # 加载模型
    data = mj.MjData(model)  # 创建数据

    # 获取关节索引（中文注释）
    jid_base_y, adr_qpos_y, adr_qvel_y = get_joint_adr("base_y")  # y 关节索引
    jid_base_roll, adr_qpos_roll, adr_qvel_roll = get_joint_adr("base_roll")  # roll 关节索引

    # 获取执行器索引（中文注释）
    aid_left = get_actuator_id("left_wheel_tau")  # 左轮力矩执行器
    aid_right = get_actuator_id("right_wheel_tau")  # 右轮力矩执行器

    # 初始化相机与渲染（中文注释）
    cam = mj.MjvCamera()  # 相机
    opt = mj.MjvOption()  # 选项
    mj.mjv_defaultCamera(cam)  # 默认相机
    mj.mjv_defaultOption(opt)  # 默认选项

    # # 绑定 XML 中的跟随相机（中文注释）
    # cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "track")  # 相机 id
    # cam.type = mj.mjtCamera.mjCAMERA_FIXED  # 固定相机
    # cam.fixedcamid = cam_id  # 使用 XML 中的相机

    # 初始化窗口（中文注释）
    glfw.init()  # 初始化 GLFW
    window = glfw.create_window(1200, 900, "MuJoCo LQR Roll 2D", None, None)  # 创建窗口
    glfw.make_context_current(window)  # 绑定上下文
    glfw.swap_interval(1)  # 开启 VSync

    # 创建场景与上下文（中文注释）
    scene = mj.MjvScene(model, maxgeom=10000)  # 场景
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染上下文

    # 绑定回调（中文注释）
    glfw.set_key_callback(window, keyboard)  # 键盘回调
    glfw.set_cursor_pos_callback(window, mouse_move)  # 鼠标移动回调
    glfw.set_mouse_button_callback(window, mouse_button)  # 鼠标按键回调
    glfw.set_scroll_callback(window, scroll)  # 滚轮回调

    # 初始化控制器（中文注释）
    init_controller()  # 计算 LQR 增益
    mj.set_mjcb_control(controller)  # 设置控制回调

    # 仿真主循环（中文注释）
    while not glfw.window_should_close(window):
        time_prev = data.time  # 上一时刻
        while data.time - time_prev < 1.0 / 60.0:  # 控制渲染频率
            mj.mj_step(model, data)  # 仿真一步
        if data.time >= simend:  # 结束条件
            break
        # 获取窗口尺寸（中文注释）
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)  # 尺寸
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)  # 视口
        # （可选）打印相机参数（中文注释）
        if print_camera_config == 1:
            print("cam.azimuth =", cam.azimuth, "; cam.elevation =", cam.elevation, "; cam.distance =", cam.distance)
            print("cam.lookat =", cam.lookat)
        # 更新场景并渲染（中文注释）
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        # 交换缓冲并处理事件（中文注释）
        glfw.swap_buffers(window)  # 交换缓冲
        glfw.poll_events()  # 处理事件

    glfw.terminate()  # 退出


if __name__ == "__main__":
    main()  # 入口