import os  # 系统路径
import numpy as np  # 数值计算
import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # GLFW 绑定
import control  # LQR 工具
import matplotlib  # 绘图配置
import matplotlib.pyplot as plt  # 绘图
from matplotlib import font_manager as fm  # 字体管理


def setup_chinese_font():  # 设置中文字体
    candidates = [  # 候选字体
        "Microsoft YaHei",  # 微软雅黑
        "SimHei",  # 黑体
        "SimSun",  # 宋体
        "NSimSun",  # 新宋体
        "FangSong",  # 仿宋
        "KaiTi",  # 楷体
        "Microsoft JhengHei",  # 微软正黑
        "Noto Sans CJK SC",  # 思源黑体
        "Source Han Sans CN",  # 思源黑体
    ]  # 字体列表
    available = {f.name for f in fm.fontManager.ttflist}  # 已有字体集合
    for name in candidates:  # 遍历候选
        if name in available:  # 找到可用字体
            matplotlib.rcParams["font.sans-serif"] = [name]  # 设置字体
            matplotlib.rcParams["axes.unicode_minus"] = False  # 负号显示
            return True, name  # 返回成功
    return False, None  # 返回失败


use_chinese_font, selected_font = setup_chinese_font()  # 初始化字体
if use_chinese_font:  # 有中文字体
    label_phi = "roll 角"  # 曲线标签
    label_u = "控制输入 u"  # 曲线标签
    label_time = "时间 (s)"  # 横轴标签
    label_phi_y = "角度 (rad)"  # 纵轴标签
    label_u_y = "u (N·m)"  # 纵轴标签
else:  # 没有中文字体
    label_phi = "roll angle"  # 英文标签
    label_u = "control input u"  # 英文标签
    label_time = "time (s)"  # 英文标签
    label_phi_y = "angle (rad)"  # 英文标签
    label_u_y = "u (N·m)"  # 英文标签


xml_path = "ip2d.xml"  # 模型路径
simend = 25.0  # 仿真总时间
settle_time = 0  # 初始稳定时间
print_camera_config = 0  # 是否打印相机参数
phi_init = 0.2# 初始小侧倾
move_start = 10.0  # 开始向 y 方向运动的时间
vy_cmd = 0.2  # 目标 y 方向速度
vy_ramp_time = 0  # 速度爬坡时间

model = None  # MuJoCo 模型
data = None  # MuJoCo 数据
cam = None  # 相机
scene = None  # 场景
context = None  # 渲染上下文
opt = None  # 可视化选项

K = None  # LQR 增益
x0 = None  # 平衡点状态（使用重心）
x0_joint = None  # 平衡点关节状态（用于线性化）
qpos0 = None  # 平衡点 qpos
qvel0 = None  # 平衡点 qvel

adr_qpos_y = None  # y 关节 qpos 索引
adr_qvel_y = None  # y 关节 qvel 索引
adr_qpos_roll = None  # roll 关节 qpos 索引
adr_qvel_roll = None  # roll 关节 qvel 索引
bid_chassis = None  # 车身 body id

aid_left = None  # 左轮执行器 id
aid_right = None  # 右轮执行器 id

t_log = []  # 时间记录
phi_log = []  # roll 记录
u_log = []  # 控制输入记录


def get_joint_adr(joint_name):  # 获取关节索引
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # 关节 id
    qpos_adr = model.jnt_qposadr[jid]  # qpos 索引
    dof_adr = model.jnt_dofadr[jid]  # qvel 索引
    return qpos_adr, dof_adr  # 返回索引


def get_actuator_id(act_name):  # 获取执行器 id
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)  # 执行器 id


def get_state_from_data(data_in):  # 读取状态
    y = data_in.qpos[adr_qpos_y]  # base_y 位置
    y_dot = data_in.qvel[adr_qvel_y]  # base_y 速度
    phi = data_in.qpos[adr_qpos_roll]  # roll 角
    phi_dot = data_in.qvel[adr_qvel_roll]  # roll 角速度
    return np.array([y, y_dot, phi, phi_dot], dtype=float)  # 状态向量


def get_joint_state_for_linearize(data_in):  # 读取关节状态（用于线性化）
    y = data_in.qpos[adr_qpos_y]  # base_y 位置
    y_dot = data_in.qvel[adr_qvel_y]  # base_y 速度
    phi = data_in.qpos[adr_qpos_roll]  # roll 角
    phi_dot = data_in.qvel[adr_qvel_roll]  # roll 角速度
    return np.array([y, y_dot, phi, phi_dot], dtype=float)  # 关节状态向量


def set_state_on_data(data_in, x_in, qpos_ref, qvel_ref):  # 写入状态
    data_in.qpos[:] = qpos_ref  # 还原参考 qpos
    data_in.qvel[:] = qvel_ref  # 还原参考 qvel
    data_in.qpos[adr_qpos_y] = x_in[0]  # 写入 y
    data_in.qvel[adr_qvel_y] = x_in[1]  # 写入 y_dot
    data_in.qpos[adr_qpos_roll] = x_in[2]  # 写入 roll
    data_in.qvel[adr_qvel_roll] = x_in[3]  # 写入 roll_dot
    mj.mj_forward(model, data_in)  # 更新前向动力学


def g_discrete(x_in, u_in, qpos_ref, qvel_ref):  # 离散一步映射
    data_lin = mj.MjData(model)  # 新数据对象
    set_state_on_data(data_lin, x_in, qpos_ref, qvel_ref)  # 设置状态
    data_lin.ctrl[aid_left] = 0.5 * u_in[0]  # 左轮力矩
    data_lin.ctrl[aid_right] = 0.5 * u_in[0]  # 右轮力矩
    mj.mj_step(model, data_lin)  # 前进一步
    return get_state_from_data(data_lin)  # 返回下一步状态


def linearize_discrete(x_eq, u_eq, qpos_ref, qvel_ref, eps=None):  # 数值线性化
    if eps is None:  # 未指定扰动
        eps = np.array([1e-4, 1e-3, 1e-4, 1e-3], dtype=float)  # 扰动大小
    n = x_eq.shape[0]  # 状态维数
    m = u_eq.shape[0]  # 输入维数
    Ad = np.zeros((n, n), dtype=float)  # 离散 A
    Bd = np.zeros((n, m), dtype=float)  # 离散 B
    x0_next = g_discrete(x_eq, u_eq, qpos_ref, qvel_ref)  # 基准步进
    for i in range(n):  # 状态扰动
        dx = np.zeros(n, dtype=float)  # 扰动向量
        dx[i] = eps[i]  # 设置扰动
        xi_next = g_discrete(x_eq + dx, u_eq, qpos_ref, qvel_ref)  # 扰动步进
        Ad[:, i] = (xi_next - x0_next) / eps[i]  # 数值差分
    for j in range(m):  # 输入扰动
        du = np.zeros(m, dtype=float)  # 输入扰动
        du[j] = 1e-2  # 输入扰动幅值
        ui_next = g_discrete(x_eq, u_eq + du, qpos_ref, qvel_ref)  # 扰动步进
        Bd[:, j] = (ui_next - x0_next) / du[j]  # 数值差分
    return Ad, Bd  # 返回线性模型


def init_controller():  # 初始化控制器
    global K, x0, x0_joint, qpos0, qvel0  # 使用全局变量
    while data.time < settle_time:  # 初始稳定
        data.ctrl[aid_left] = 0.0  # 左轮无力矩
        data.ctrl[aid_right] = 0.0  # 右轮无力矩
        mj.mj_step(model, data)  # 仿真步进
    qpos0 = data.qpos.copy()  # 保存 qpos
    qvel0 = data.qvel.copy()  # 保存 qvel
    x0 = get_state_from_data(data)  # 平衡点状态（关节 y/roll）
    x0_joint = get_joint_state_for_linearize(data)  # 平衡点关节状态
    u0 = np.array([0.0], dtype=float)  # 平衡输入
    Ad, Bd = linearize_discrete(x0, u0, qpos0, qvel0)  # 线性化（关节扰动）
    Q = np.diag([200.0, 20.0, 2000.0, 200.0])  # 仅约束 roll/roll_dot
    R = np.diag([100])  # 输入权重
    K, S, E = control.dlqr(Ad, Bd, Q, R)  # 离散 LQR
    print("x0 =", x0)  # 输出平衡点
    print("LQR K =", K)  # 输出增益gei


def controller(model_in, data_in):  # 控制回调
    x = get_state_from_data(data_in)  # 读取状态
    if data_in.time < move_start:  # 先平衡
        y_ref = x0[0]  # y 参考（保持初始）
        ydot_ref = 0.0  # y 速度参考
    else:  # 再向 y 方向匀速运动
        t_run = data_in.time - move_start  # 运动阶段时间
        ramp = min(t_run / max(vy_ramp_time, 1e-6), 1.0)  # 速度爬坡比例
        ydot_ref = vy_cmd * ramp  # 速度参考平滑上升
        y_ref = x0[0] + 0.5 * vy_cmd * min(t_run, vy_ramp_time) * ramp + vy_cmd * max(t_run - vy_ramp_time, 0.0)  # 位置参考
    roll_ref = 0.0  # roll 参考（直立）
    roll_dot_ref = 0.0  # roll 角速度参考
    x_ref = np.array([y_ref, ydot_ref, roll_ref, roll_dot_ref], dtype=float)  # 参考状态
    e = x - x_ref  # 跟踪误差
    u = -K @ e  # LQR 控制
    u = np.clip(u, -200.0, 200.0)  # 力矩限幅
    data_in.ctrl[aid_left] = 0.5 * u[0]  # 左轮力矩
    data_in.ctrl[aid_right] = 0.5 * u[0]  # 右轮力矩
    t_log.append(data_in.time)  # 记录时间
    phi_log.append(x[2])  # 记录 roll
    u_log.append(u[0])  # 记录控制量


button_left = False  # 鼠标左键
button_middle = False  # 鼠标中键
button_right = False  # 鼠标右键
lastx = 0  # 鼠标 x
lasty = 0  # 鼠标 y


def keyboard(window, key, scancode, act, mods):  # 键盘回调
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:  # 按下退格
        mj.mj_resetData(model, data)  # 重置数据
        mj.mj_forward(model, data)  # 前向更新


def mouse_button(window, button, act, mods):  # 鼠标按键回调
    global button_left, button_middle, button_right  # 使用全局变量
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)  # 左键
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)  # 中键
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)  # 右键
    glfw.get_cursor_pos(window)  # 更新鼠标位置


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
    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)  # 更新相机


def scroll(window, xoffset, yoffset):  # 滚轮回调
    action = mj.mjtMouse.mjMOUSE_ZOOM  # 缩放
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)  # 更新相机


def main():  # 主函数
    global model, data, cam, scene, context, opt  # 全局对象
    global adr_qpos_y, adr_qvel_y, adr_qpos_roll, adr_qvel_roll, bid_chassis  # 索引
    global aid_left, aid_right  # 执行器 id
    dirname = os.path.dirname(__file__)  # 当前目录
    abspath = os.path.join(dirname, xml_path)  # 模型路径
    model = mj.MjModel.from_xml_path(abspath)  # 加载模型
    data = mj.MjData(model)  # 创建数据
    adr_qpos_y, adr_qvel_y = get_joint_adr("base_y")  # y 索引
    adr_qpos_roll, adr_qvel_roll = get_joint_adr("base_roll")  # roll 索引
    bid_chassis = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "chassis")  # 车身 body id
    aid_left = get_actuator_id("left_wheel_tau")  # 左轮执行器
    aid_right = get_actuator_id("right_wheel_tau")  # 右轮执行器
    cam = mj.MjvCamera()  # 相机
    opt = mj.MjvOption()  # 选项
    mj.mjv_defaultCamera(cam)  # 默认相机
    mj.mjv_defaultOption(opt)  # 默认选项
    glfw.init()  # 初始化 GLFW
    window = glfw.create_window(1200, 900, "MuJoCo LQR Roll 2D", None, None)  # 创建窗口
    glfw.make_context_current(window)  # 绑定上下文
    glfw.swap_interval(1)  # 垂直同步
    scene = mj.MjvScene(model, maxgeom=10000)  # 场景
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染
    glfw.set_key_callback(window, keyboard)  # 键盘回调
    glfw.set_cursor_pos_callback(window, mouse_move)  # 鼠标移动回调
    glfw.set_mouse_button_callback(window, mouse_button)  # 鼠标按键回调
    glfw.set_scroll_callback(window, scroll)  # 滚轮回调
    init_controller()  # 初始化控制器
    data.qpos[adr_qpos_roll] = data.qpos[adr_qpos_roll] + phi_init  # 初始侧倾
    mj.mj_forward(model, data)  # 前向更新
    mj.set_mjcb_control(controller)  # 设置控制回调
    while not glfw.window_should_close(window):  # 主循环
        time_prev = data.time  # 上一时刻
        while data.time - time_prev < 1.0 / 1000.0:  # 控制渲染频率
            mj.mj_step(model, data)  # 仿真步进
        if data.time >= simend:  # 结束条件
            break  # 退出循环
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)  # 窗口尺寸
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)  # 视口
        if print_camera_config == 1:  # 打印相机参数
            print("cam.azimuth =", cam.azimuth, "; cam.elevation =", cam.elevation, "; cam.distance =", cam.distance)  # 打印
            print("cam.lookat =", cam.lookat)  # 打印
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)  # 更新场景
        mj.mjr_render(viewport, scene, context)  # 渲染
        glfw.swap_buffers(window)  # 交换缓冲
        glfw.poll_events()  # 处理事件
    glfw.terminate()  # 释放资源
    if len(t_log) > 1:  # 有数据才绘图
        fig = plt.figure(figsize=(10, 6))  # 创建图
        ax1 = fig.add_subplot(2, 1, 1)  # 子图1
        ax1.plot(t_log, phi_log, label=label_phi)  # 画 roll
        ax1.set_ylabel(label_phi_y)  # y 标签
        ax1.grid(True)  # 网格
        ax1.legend()  # 图例
        ax2 = fig.add_subplot(2, 1, 2)  # 子图2
        ax2.plot(t_log, u_log, label=label_u)  # 画 u
        ax2.set_xlabel(label_time)  # x 标签
        ax2.set_ylabel(label_u_y)  # y 标签
        ax2.grid(True)  # 网格
        ax2.legend()  # 图例
        out_png = os.path.join(os.path.dirname(__file__), "lqr_roll2d_plot.png")  # 输出路径
        plt.tight_layout()  # 紧凑布局
        plt.savefig(out_png, dpi=200)  # 保存图片
        plt.close(fig)  # 关闭图


if __name__ == "__main__":  # 入口判断
    main()  # 运行主程序
