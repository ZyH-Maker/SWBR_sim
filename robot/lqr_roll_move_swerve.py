import os  # 系统路径
import math  # 数学函数
import numpy as np  # 数值计算
import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # GLFW 绑定

xml_path = "ip2d.xml"  # 模型文件
simend = 1200.0  # 仿真时长
move_start = 0  # 平衡结束后开始运动时间

# ========= 速度指令（世界坐标）=========
vx_cmd = 0.0  # 期望 x 方向速度
vy_cmd = 0.0  # 期望 y 方向速度
wz_cmd = 0.0  # 期望 yaw 角速度
v_step = 0.1  # 速度增量
w_step = 0.5  # 角速度增量
v_max = 10  # 速度上限
w_max = 1.0  # 角速度上限


# ========= 平衡层（LQR）=========
K = np.array([[-13.78243357, -70.62971897, 242.32442576, 23.44076849]])  # 从 lqr_roll2d.py 填入

# ========= 速度外环（生成力）=========
kv_xy = 50 # 速度比例增益
fx_max = 200.0  # x 方向力限幅
fy_max = 200.0  # y 方向力限幅

# ========= 力矩限幅 =========
tau_max = 200.0  # 力矩限幅

# ========= 全局对象 =========
model = None  # MuJoCo 模型
data = None  # MuJoCo 数据
cam = None  # 相机
scene = None  # 场景
context = None  # 渲染上下文
opt = None  # 可视化选项
x0 = None  # 平衡参考状态
last_print_time = -1.0  # 上次打印时间
y_ref_base = 0.0  # y 参考基准
vy_ref_last = 0.0  # 上一次 vy 命令
t_ref_start = 0.0  # 参考起始时间

# ========= 索引 =========
aid_lw = -1  # 左轮力矩执行器 id
aid_rw = -1  # 右轮力矩执行器 id
aid_ls = -1  # 左舵角执行器 id
aid_rs = -1  # 右舵角执行器 id
adr_qpos_y = -1  # base_y 位置索引
adr_qvel_x = -1  # base_x 速度索引
adr_qvel_y = -1  # base_y 速度索引
adr_qpos_roll = -1  # roll 位置索引
adr_qvel_roll = -1  # roll 速度索引
adr_qpos_ls = -1  # 左舵角位置索引
adr_qpos_rs = -1  # 右舵角位置索引

# ========= 结构参数 =========
wheel_radius = 0.05  # 轮子半径
# steer_offset = -math.pi / 2.0  # 舵角偏置（模型初始旋转了 90 度）
x_front = 0.0  # 前轮模块 x
y_front = 0.0  # 前轮模块 y
x_rear = 0.0  # 后轮模块 x
y_rear = 0.0  # 后轮模块 y

# ========= 鼠标 =========
button_left = False  # 鼠标左键
button_middle = False  # 鼠标中键
button_right = False  # 鼠标右键
lastx = 0  # 鼠标 x
lasty = 0  # 鼠标 y
prev_steer_f = 0.0  # 前轮上一帧舵角
prev_steer_r = 0.0  # 后轮上一帧舵角


def clamp(v, vmin, vmax):  # 限幅
    return max(vmin, min(vmax, v))  # 返回限幅值


def wrap_to_pi(a):  # 角度归一化
    while a > math.pi:  # 大于 pi
        a -= 2.0 * math.pi  # 减 2pi
    while a < -math.pi:  # 小于 -pi
        a += 2.0 * math.pi  # 加 2pi
    return a  # 返回角度

def clamp_steer_90(a):  # 舵角折叠到 [-90°, 90°]
    a = wrap_to_pi(a)
    # half_pi = math.pi / 2.0
    # if a > half_pi:
    #     a -= math.pi
    # elif a < -half_pi:
    #     a += math.pi
    return a


def unwrap_angle(prev, curr):  # 角度连续化
    prev = float(prev)  # 上一帧角度
    curr = wrap_to_pi(float(curr))  # 当前角度归一化
    delta = wrap_to_pi(curr - prev)  # 计算最短差
    return prev + delta  # 返回连续角度


def get_state():  # 读取平衡状态
    y = float(data.qpos[adr_qpos_y])  # base_y 位置
    y_dot = float(data.qvel[adr_qvel_y])  # base_y 速度
    roll = float(data.qpos[adr_qpos_roll])  # roll 角
    roll_dot = float(data.qvel[adr_qvel_roll])  # roll 角速度
    return np.array([y, y_dot, roll, roll_dot], dtype=float)  # 状态向量


def get_body_vel():  # 读取平动速度
    vx = float(data.qvel[adr_qvel_x])  # base_x 速度
    vy = float(data.qvel[adr_qvel_y])  # base_y 速度
    return vx, vy  # 返回速度


def steer_solve(vx, vy, wz, x_i, y_i):  # 舵轮解算
    vix = vx - wz * y_i  # 模块 x 速度
    viy = vy + wz * x_i  # 模块 y 速度
    steer = math.atan2(viy, vix)  # 舵角
    speed = math.hypot(vix, viy)  # 线速度
    return steer, speed  # 返回舵角与线速度


def keyboard(window, key, scancode, act, mods):  # 键盘回调
    global vx_cmd, vy_cmd, wz_cmd  # 使用全局命令
    if act not in (glfw.PRESS, glfw.REPEAT):  # 只处理按下/长按
        return  # 直接返回
    if key == glfw.KEY_BACKSPACE:  # 复位
        mj.mj_resetData(model, data)  # 重置数据
        mj.mj_forward(model, data)  # 前向更新
        return  # 直接返回
    if key == glfw.KEY_UP:  # x+
        vx_cmd += v_step  # 增加 x
    elif key == glfw.KEY_DOWN:  # x-
        vx_cmd -= v_step  # 减少 x
    elif key == glfw.KEY_LEFT:  # y+
        vy_cmd += v_step  # 增加 y
    elif key == glfw.KEY_RIGHT:  # y-
        vy_cmd -= v_step  # 减少 y
    elif key == glfw.KEY_Q:  # wz+
        wz_cmd += w_step  # 增加 yaw
    elif key == glfw.KEY_E:  # wz-
        wz_cmd -= w_step  # 减少 yaw
    elif key == glfw.KEY_SPACE:  # 清零
        vx_cmd = 0.0  # 清零 x
        vy_cmd = 0.0  # 清零 y
        wz_cmd = 0.0  # 清零 yaw
    vx_cmd = clamp(vx_cmd, -v_max, v_max)  # 限幅
    vy_cmd = clamp(vy_cmd, -v_max, v_max)  # 限幅
    wz_cmd = clamp(wz_cmd, -w_max, w_max)  # 限幅


def mouse_button(window, button, act, mods):  # 鼠标按键
    global button_left, button_middle, button_right, lastx, lasty  # 使用全局变量
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)  # 左键
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)  # 中键
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)  # 右键
    lastx, lasty = glfw.get_cursor_pos(window)  # 记录位置


def mouse_move(window, xpos, ypos):  # 鼠标移动
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


def scroll(window, xoffset, yoffset):  # 滚轮
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)  # 更新相机


def init_ids():  # 初始化索引
    global aid_lw, aid_rw, aid_ls, aid_rs  # 执行器 id
    global adr_qpos_y, adr_qvel_x, adr_qvel_y, adr_qpos_roll, adr_qvel_roll, adr_qpos_ls, adr_qpos_rs  # 关节索引
    global wheel_radius, x_front, y_front, x_rear, y_rear  # 结构参数
    aid_lw = model.actuator("left_wheel_tau").id  # 左轮力矩执行器
    aid_rw = model.actuator("right_wheel_tau").id  # 右轮力矩执行器
    aid_ls = model.actuator("left_steer_pos").id  # 左舵角执行器
    aid_rs = model.actuator("right_steer_pos").id  # 右舵角执行器
    jid_y = model.joint("base_y").id  # base_y 关节 id
    jid_x = model.joint("base_x").id  # base_x 关节 id
    jid_roll = model.joint("base_roll").id  # roll 关节 id
    jid_ls = model.joint("left_steer").id  # 左舵角关节 id
    jid_rs = model.joint("right_steer").id  # 右舵角关节 id
    adr_qpos_y = model.jnt_qposadr[jid_y]  # base_y 位置索引
    adr_qvel_x = model.jnt_dofadr[jid_x]  # base_x 速度索引
    adr_qvel_y = model.jnt_dofadr[jid_y]  # base_y 速度索引
    adr_qpos_roll = model.jnt_qposadr[jid_roll]  # roll 位置索引
    adr_qpos_ls = model.jnt_qposadr[jid_ls]  # 左舵角位置索引
    adr_qpos_rs = model.jnt_qposadr[jid_rs]  # 右舵角位置索引
    adr_qvel_roll = model.jnt_dofadr[jid_roll]  # roll 速度索引
    wheel_radius = float(model.geom("left_wheel_geom").size[0])  # 轮子半径
    x_front = float(model.body("left_module").pos[0])  # 前轮模块 x
    y_front = float(model.body("left_module").pos[1])  # 前轮模块 y
    x_rear = float(model.body("right_module").pos[0])  # 后轮模块 x
    y_rear = float(model.body("right_module").pos[1])  # 后轮模块 y

import math

def shortest_angle(prev, new):
    """
    让 new 相对于 prev 取最近的等价角度
    返回连续角度，不会出现 ±π 跳变
    """
    return prev + math.atan2(
        math.sin(new - prev),
        math.cos(new - prev)
    )



def controller(model_in, data_in):  # 控制回调
    global x0, last_print_time, y_ref_base, vy_ref_last, t_ref_start, prev_steer_f, prev_steer_r  # 使用缓存
    x = get_state()  # 读取状态
    if x0 is None:  # 首次进入控制器
        x0 = x.copy()  # 记录初始平衡点
        y_ref_base = x0[0]  # 初始化 y 参考基准
        vy_ref_last = vy_cmd  # 初始化速度
        t_ref_start = data_in.time  # 初始化参考时间
    if data_in.time < move_start:  # 先平衡
        y_ref = x0[0]  # y 参考
        ydot_ref = 0.0  # y 速度参考
    else:  # 再向 y 方向匀速运动
        if abs(vy_cmd - vy_ref_last) > 1e-6:  # 速度指令变化
            y_ref_base = y_ref_base + vy_ref_last * (data_in.time - t_ref_start)  # 更新参考基准
            t_ref_start = data_in.time  # 更新起始时间
            vy_ref_last = vy_cmd  # 更新速度
        ydot_ref = vy_cmd  # y 速度参考
        y_ref = y_ref_base + vy_cmd * (data_in.time - t_ref_start)  # y 位置参考
        #print(f"y_ref={y_ref:.3f} ydot_ref={ydot_ref:.3f}")  # 调试打印
    roll_ref = 0.0  # roll 参考
    roll_dot_ref = 0.0  # roll 角速度参考
    x_ref = np.array([y_ref, ydot_ref, roll_ref, roll_dot_ref], dtype=float)  # 参考状态
    e = x - x_ref  # 跟踪误差
    u_bal = -K @ e  # LQR 输出
    u_bal = float(u_bal[0])  # 转标量
    fy_bal = u_bal / max(wheel_radius, 1e-6)  # 平衡侧向力
    # print(f"fy_bal={fy_bal:.3f}")  # 调试打印
    # fy_bal = clamp(fy_bal, -fy_max, fy_max)  # 限幅
    vx, vy = get_body_vel()  # 当前速度
    vx_use = vx_cmd  # x 指令
    vy_use = vy_cmd  # y 指令
    vx_meas = vx  # x 测量
    vy_meas = vy  # y 测量
    fx_track = clamp(kv_xy * (vx_use - vx_meas), -fx_max, fx_max)  # x 速度跟踪力
    # fy_track = clamp(kv_xy * (vy_use - vy_meas), -fy_max, fy_max)  # y 速度跟踪力
    fx_cmd = fx_track  # 合力 x 分量
    fy_cmd =  fy_bal  # 合力 y 分量（叠加平衡）
    cmd_mag = abs(vx_use) + abs(vy_use) + abs(wz_cmd)  # 指令幅值
    if data_in.time < move_start or cmd_mag < 1e-6:  # 平衡阶段或无指令
        steer_f = 0.0  # 舵角保持为模型默认方向
        steer_r = 0.0  # 舵角保持为模型默认方向
    else:  # 进入舵轮解算
        # steer_f, _ = steer_solve(vx_use, vy_use, wz_cmd, x_front, y_front)  # 前轮解算
        # steer_r, _ = steer_solve(vx_use, vy_use, wz_cmd, x_rear, y_rear)  # 后轮解算
        # 1️⃣ 计算目标角度
        target_angle = math.atan2(fx_cmd, fy_cmd)
        # 2️⃣ 前后轮同向
        steer_f = shortest_angle(prev_steer_f, target_angle)
        steer_r = shortest_angle(prev_steer_r, target_angle)

        # 3️⃣ 更新历史
        prev_steer_f = steer_f
        prev_steer_r = steer_r
        print(f"fx_cmd={fx_cmd:.3f} fy_cmd={fy_cmd:.3f} steer_r={steer_r:.3f} steer_r_cmd={wrap_to_pi(steer_r):.3f}")  # 调试打印
        # steer_f = wrap_to_pi(steer_f + steer_offset)  # 前轮舵角偏置修正
        # steer_r = wrap_to_pi(steer_r + steer_offset)  # 后轮舵角偏置修正
    data_in.ctrl[aid_ls] = float(clamp_steer_90(steer_f))  # 左舵角目标
    data_in.ctrl[aid_rs] = float(clamp_steer_90(steer_r))  # 右舵角目标
    f_mag = math.hypot(fx_cmd, fy_cmd)  # 合力大小
    if fy_bal < 1e-6:  # 无平衡力
        tau_each = 0.5 * u_bal  # 平衡力矩
    else:
        tau_each = 0.5 * f_mag * wheel_radius  # 统一通道力矩
    data_in.ctrl[aid_lw] = float(clamp(tau_each, -tau_max, tau_max))  # 左轮力矩输出
    data_in.ctrl[aid_rw] = float(clamp(tau_each, -tau_max, tau_max))  # 右轮力矩输出
    # if data_in.time < 1e-3:  # 初始时刻打印舵角
    #     try:
    #         cur_ls = float(data.qpos[adr_qpos_ls])  # 左舵角当前值
    #         cur_rs = float(data.qpos[adr_qpos_rs])  # 右舵角当前值
    #         print(f"[t={data_in.time:.3f}] 初始舵角 left={cur_ls:.3f} rad, right={cur_rs:.3f} rad")  # 打印
    #     except Exception:
    #         pass  # 忽略异常
    if data_in.time - last_print_time >= 0.5:  # 每 0.5 秒打印一次
        try:
            cur_ls = float(data.qpos[adr_qpos_ls])  # 左舵角当前值
            cur_rs = float(data.qpos[adr_qpos_rs])  # 右舵角当前值
        except Exception:
            cur_ls = float("nan")
            cur_rs = float("nan")
        print(
            f"[t={data_in.time:.2f}] steer_cmd_f={steer_f:.3f} steer_cmd_r={steer_r:.3f} "
            f"steer_qpos_l={cur_ls:.3f} steer_qpos_r={cur_rs:.3f} "
            f"u_bal={u_bal:.3f} tau_each={tau_each:.3f}"
        )  # 打印调试
        last_print_time = data_in.time  # 更新打印时间


def main():  # 主函数
    global model, data, cam, scene, context, opt  # 全局对象
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
    window = glfw.create_window(1200, 900, "LQR Balance + Swerve", None, None)  # 创建窗口
    glfw.make_context_current(window)  # 绑定上下文
    glfw.swap_interval(1)  # 垂直同步
    scene = mj.MjvScene(model, maxgeom=10000)  # 场景
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)  # 渲染
    glfw.set_key_callback(window, keyboard)  # 键盘回调
    glfw.set_cursor_pos_callback(window, mouse_move)  # 鼠标移动回调
    glfw.set_mouse_button_callback(window, mouse_button)  # 鼠标按键回调
    glfw.set_scroll_callback(window, scroll)  # 滚轮回调
    mj.set_mjcb_control(controller)  # 注册控制器
    cam.azimuth = 90.0  # 相机方位
    cam.elevation = -20.0  # 相机俯仰
    cam.distance = 2.0  # 相机距离
    t0 = data.time  # 起始时间
    while not glfw.window_should_close(window) and data.time - t0 < simend:  # 主循环
        time_prev = data.time  # 上一时刻
        while data.time - time_prev < 1.0 / 1000.0:  # 60Hz 渲染
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
