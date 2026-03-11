import os  # 系统路径
import numpy as np  # 数值计算
import mujoco as mj  # MuJoCo 主库
from mujoco.glfw import glfw  # GLFW 绑定
import control  # LQR 工具
import matplotlib  # 绘图配置
import matplotlib.pyplot as plt  # 绘图
from matplotlib import font_manager as fm  # 字体管理
import scipy.linalg  # 线性代数
from scipy.interpolate import CubicSpline

# 全局变量：用于存储拟合好的 LPV-LQR 插值器
K_interpolator = None

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
    label_x_y = "x 位置"  # 曲线标签
    label_v_y = "y 速度"  # 曲线标签
    label_phi_y = "角度 (rad)"  # 纵轴标签
    label_w_y = "角速度 (rad/s)"  # 纵轴标签
    label_u_y = "u (N·m)"  # 纵轴标签
else:  # 没有中文字体
    label_phi = "roll angle"  # 英文标签
    label_u = "control input u"  # 英文标签
    label_time = "time (s)"  # 英文标签
    label_x_y = "x position"  # 英文标签    
    label_v_y = "y velocity"  # 英文标签
    label_phi_y = "angle (rad)"  # 英文标签
    label_w_y = "angular velocity (rad/s)"  # 英文标签
    label_u_y = "u (N·m)"  # 英文标签


xml_path = "ip2d.xml"  # 模型路径
simend = 15.0  # 仿真总时间
simspeed = 1.0  # 仿真速度
settle_time = 0  # 初始稳定时间
print_camera_config = 0  # 是否打印相机参数
phi_init = 0.75  # 初始小侧倾
move_start = 5.0  # 开始向 y 方向运动的时间
vy_cmd = 2.5  # 目标 y 方向速度
vy_ramp_time = 0  # 速度爬坡时间
move_end = 10.0  # 结束向 y 方向运动的时间 

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
Q = None  # 状态权重矩阵
R = None  # 输入权重矩阵

adr_qpos_y = None  # y 关节 qpos 索引
adr_qvel_y = None  # y 关节 qvel 索引
adr_qpos_roll = None  # roll 关节 qpos 索引
adr_qvel_roll = None  # roll 关节 qvel 索引
bid_chassis = None  # 车身 body id

aid_left = None  # 左轮执行器 id
aid_right = None  # 右轮执行器 id

t_log = []  # 时间记录
x_log = []  # 状态记录
v_log = []  # 速度记录（车身质心）
v_wheel_log = []  # 速度记录（轮子接触点）
phi_log = []  # roll 记录
w_log = []  # 角速度记录
u_log = []  # 控制输入记录
y_ref_log = []  # y参考记录
ydot_ref_log = []  # y速度参考记录
e_y_log = []  # y误差记录
e_ydot_log = []  # y_dot误差记录
e_phi_log = []  # phi误差记录
e_phidot_log = []  # phi_dot误差记录
u_y_log = []  # u由y误差产生的分量
u_ydot_log = []  # u由y_dot误差产生的分量
u_phi_log = []  # u由phi误差产生的分量
u_phidot_log = []  # u由phi_dot误差产生的分量


def get_joint_adr(joint_name):  # 获取关节索引
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)  # 关节 id
    qpos_adr = model.jnt_qposadr[jid]  # qpos 索引
    dof_adr = model.jnt_dofadr[jid]  # qvel 索引
    return qpos_adr, dof_adr  # 返回索引


def get_actuator_id(act_name):  # 获取执行器 id
    return mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, act_name)  # 执行器 id


h_cm = 0.31  # 车身质心高度（从模型文件中获取）


def get_state_from_data(data_in):  # 读取状态（车身质心坐标系）
    y_wheel = data_in.qpos[adr_qpos_y]  # base_y 位置（轮子接触点）
    y_dot_wheel = data_in.qvel[adr_qvel_y]  # base_y 速度（轮子接触点）
    phi = data_in.qpos[adr_qpos_roll]  # roll 角
    phi_dot = data_in.qvel[adr_qvel_roll]  # roll 角速度
    # 计算车身质心的位置和速度
    # roll正方向绕x轴，车身向y负方向倾斜
    y_cm = y_wheel - h_cm * np.sin(phi)  # 车身质心y位置
    y_dot_cm = y_dot_wheel - h_cm * np.cos(phi) * phi_dot  # 车身质心y速度
    return np.array([y_cm, y_dot_cm, phi, phi_dot], dtype=float)  # 状态向量


def get_joint_state_for_linearize(data_in):  # 读取关节状态（用于线性化，保持轮子坐标系）
    y = data_in.qpos[adr_qpos_y]  # base_y 位置
    y_dot = data_in.qvel[adr_qvel_y]  # base_y 速度
    phi = data_in.qpos[adr_qpos_roll]  # roll 角
    phi_dot = data_in.qvel[adr_qvel_roll]  # roll 角速度
    return np.array([y, y_dot, phi, phi_dot], dtype=float)  # 关节状态向量


def set_state_on_data(data_in, x_in, qpos_ref, qvel_ref):  # 写入状态（x_in是车身质心状态）
    data_in.qpos[:] = qpos_ref  # 还原参考 qpos
    data_in.qvel[:] = qvel_ref  # 还原参考 qvel
    # 从车身质心状态转换回轮子接触点状态
    y_cm = x_in[0]  # 车身质心y位置
    y_dot_cm = x_in[1]  # 车身质心y速度
    phi = x_in[2]  # roll角
    phi_dot = x_in[3]  # roll角速度
    # 逆变换：从车身质心状态计算轮子接触点状态
    # roll正方向绕x轴，车身向y负方向倾斜
    y_wheel = y_cm + h_cm * np.sin(phi)  # 轮子接触点y位置
    y_dot_wheel = y_dot_cm + h_cm * np.cos(phi) * phi_dot  # 轮子接触点y速度
    data_in.qpos[adr_qpos_y] = y_wheel  # 写入 y
    data_in.qvel[adr_qvel_y] = y_dot_wheel  # 写入 y_dot
    data_in.qpos[adr_qpos_roll] = phi  # 写入 roll
    data_in.qvel[adr_qvel_roll] = phi_dot  # 写入 roll_dot
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
        eps = np.array([1e-5, 1e-4, 1e-5, 1e-4], dtype=float)  # 扰动大小
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
        du[j] = 1e-3  # 输入扰动幅值
        ui_next = g_discrete(x_eq, u_eq + du, qpos_ref, qvel_ref)  # 扰动步进
        Bd[:, j] = (ui_next - x0_next) / du[j]  # 数值差分
    print(Ad)
    print(Bd)
    return Ad, Bd  # 返回线性模型


def linearize_discrete_central(x_eq, u_eq, qpos_ref, qvel_ref, eps=None):
    """
    使用中心有限差分法获取离散状态空间模型 (A, B)
    """
    n = x_eq.shape[0]
    m = u_eq.shape[0]
    
    # 若未指定，提供默认的合理扰动幅值
    if eps is None:
        eps_x = np.full(n, 1e-5, dtype=float) # 状态扰动
        eps_u = np.full(m, 1e-4, dtype=float) # 输入扰动
    else:
        eps_x = eps[:n]
        eps_u = eps[n:] if len(eps) > n else np.full(m, 1e-3, dtype=float)

    Ad = np.zeros((n, n), dtype=float)
    Bd = np.zeros((n, m), dtype=float)

    # 1. 计算状态矩阵 A
    for i in range(n):
        dx = np.zeros(n, dtype=float)
        dx[i] = eps_x[i]
        
        # 正向扰动
        x_plus = g_discrete(x_eq + dx, u_eq, qpos_ref, qvel_ref)
        # 负向扰动
        x_minus = g_discrete(x_eq - dx, u_eq, qpos_ref, qvel_ref)
        
        # 中心差分公式
        Ad[:, i] = (x_plus - x_minus) / (2.0 * eps_x[i])

    # 2. 计算输入矩阵 B
    for j in range(m):
        du = np.zeros(m, dtype=float)
        du[j] = eps_u[j]
        
        # 正向扰动
        u_plus = g_discrete(x_eq, u_eq + du, qpos_ref, qvel_ref)
        # 负向扰动
        u_minus = g_discrete(x_eq, u_eq - du, qpos_ref, qvel_ref)
        
        # 中心差分公式
        Bd[:, j] = (u_plus - u_minus) / (2.0 * eps_u[j])
    print(Ad)
    print(Bd)
    return Ad, Bd


def init_LPV_controller():  # 初始化控制器
    global K_interpolator, x0, x0_joint, qpos0, qvel0, Q, R  # 使用全局变量
    
    while data.time < settle_time:  # 初始稳定
        data.ctrl[aid_left] = 0.0  # 左轮无力矩
        data.ctrl[aid_right] = 0.0  # 右轮无力矩
        mj.mj_step(model, data)  # 仿真步进
        
    qpos0 = data.qpos.copy()  # 保存 qpos
    qvel0 = data.qvel.copy()  # 保存 qvel
    
    x0 = get_state_from_data(data)  # 获取基准状态
    x0[2] = 0.0  # 保证初始设定 roll 角为 0
    
    Q = np.diag([120000.0, 3000000.0, 120000.0, 220000.0])  # 进一步增大权重
    R = np.diag([0.2])  # 减小R允许更大控制力
    u0 = np.array([0.0], dtype=float)  # 平衡输入
    
    print("开始进行 LPV 多点线性化与插值拟合...")
    
    # 1. 定义 roll 角的采样网格 (例如从 -60度 到 +60度，采样 31 个点)
    # 取决于你的需求，采样点越密拟合越好
    roll_grid = np.linspace(-np.radians(60), np.radians(60), 31)
    K_list = []
    
    for roll in roll_grid:
        # 构造当前角度下的瞬间状态
        x_eq = x0.copy()
        x_eq[2] = roll  # 修改 roll 角
        x_eq[3] = 0.0   # roll 角速度设为 0
        
        # 【极其重要的注意点】：
        # 你的 linearize_discrete_central 接收 qpos0 和 qvel0。
        # 如果你的内部函数 g_discrete 是强依赖于 x_eq 来覆盖并更新仿真状态的，那么下面这行没问题。
        # 但如果 g_discrete 内部是从 qpos0 读取 roll 角的，你必须在这里同步修改 qpos0 里的 roll 关节值！
        Ad, Bd = linearize_discrete_central(x_eq, u0, qpos0, qvel0)  
        
        try:
            # 求解 DARE 并计算 K
            P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R) 
            K_i = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
            K_list.append(K_i[0])  # 取出一维数组 [k1, k2, k3, k4] 存入列表
        except Exception as e:
            print(f"在 roll = {np.degrees(roll):.2f}° 处求解失败: {e}")
            # 容错机制：如果由于物理极限在极端角度求解失败，用上一个合法的 K 矩阵填补
            if len(K_list) > 0:
                K_list.append(K_list[-1])
            else:
                K_list.append(np.zeros(4))
                
    # 2. 生成三次样条插值器
    K_array = np.array(K_list)
    K_interpolator = CubicSpline(roll_grid, K_array)
    
    print("x0 =", x0)  # 输出平衡点
    print("LPV 控制器初始化完成，K 矩阵随 roll 角变化的插值器已生成！")
    print("K(roll = 0°) =", K_array[15])  # 输出 roll = 0° 时的 K 矩阵
    print(K_interpolator)
    
    # 3. 绘制K系数关于倾角的图像
    plot_K_vs_roll(roll_grid, K_array, K_interpolator)


def plot_K_vs_roll(roll_grid, K_array, K_interpolator):
    """绘制K的四个系数关于倾角的图像"""
    # 生成更密集的插值点用于绘制平滑曲线
    roll_fine = np.linspace(roll_grid[0], roll_grid[-1], 200)
    K_fine = K_interpolator(roll_fine)
    
    # 转换为角度方便阅读
    roll_grid_deg = np.degrees(roll_grid)
    roll_fine_deg = np.degrees(roll_fine)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    labels = ['$K_y$', '$K_{\dot{y}}$', '$K_{\\phi}$', '$K_{\dot{\\phi}}$']
    colors = ['b', 'g', 'r', 'm']
    
    for i, (ax, label, color) in enumerate(zip(axes.flatten(), labels, colors)):
        # 绘制原始采样点
        ax.plot(roll_grid_deg, K_array[:, i], 'o', markersize=4, color=color, label='采样点')
        # 绘制三次样条插值曲线
        ax.plot(roll_fine_deg, K_fine[:, i], '-', color=color, label='三次样条插值')
        ax.set_xlabel('roll 角 (度)')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs roll 角')
        ax.grid(True)
        ax.legend()
    
    plt.suptitle('LQR 增益系数 K 随 roll 角变化曲线', fontsize=14)
    plt.tight_layout()
    
    out_png = os.path.join(os.path.dirname(__file__), "lqr_K_vs_roll.png")
    plt.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"K系数图像已保存到: {out_png}")

def LPV_controller(model_in, data_in):  # 控制回调
    global K_interpolator, x0  # 确保能访问到插值器和初始状态
    
    x = get_state_from_data(data_in)  # 读取状态
    
    # --- 生成参考轨迹 ---
    if data_in.time < move_start:  # 先平衡
        y_ref = x0[0]  # y 参考（保持初始）
        ydot_ref = 0.0  # y 速度参考
    elif data_in.time <= move_end:  # 再向 y 方向匀速运动
        t_run = data_in.time - move_start  # 运动阶段时间
        y_ref = x0[0] + vy_cmd * t_run  # 位置参考
        ydot_ref = vy_cmd  # 速度参考
    else:
        y_ref = x0[0] + vy_cmd * (move_end - move_start)  # y 参考
        ydot_ref = 0.0  # y 速度参考
        
    roll_ref = 0.0  # roll 参考（直立）
    roll_dot_ref = 0.0  # roll 角速度参考
    x_ref = np.array([y_ref, ydot_ref, roll_ref, roll_dot_ref], dtype=float)  # 参考状态
    e = x - x_ref  # 跟踪误差
    
    # --- 核心修改：LPV 实时查表计算 K ---
    current_roll = x[2]  # 获取当前的真实 roll 角
    
    # 【安全限幅】：极其重要！
    # 防止系统受干扰 roll 角超过 40 度时，三次样条发生不可控的向外推断 (Extrapolation)
    roll_clip = np.clip(current_roll, -np.radians(58), np.radians(58))
    
    # 瞬间查询插值器，得到当前姿态最完美的反馈增益
    K_current = K_interpolator(roll_clip)
    print("K_current =", K_current)
    # print("e =", e)
    
    # LQR 控制
    u = -K_current @ e  
    
    # 力矩限幅
    u = np.clip(u, -16.0, 16.0) 
    # print("u =", u)
    
    data_in.ctrl[aid_left] = 0.5 * u  # 左轮力矩
    data_in.ctrl[aid_right] = 0.5 * u  # 右轮力矩

    t_log.append(data_in.time)  # 记录时间
    x_log.append(x[0])  # 记录状态
    v_log.append(x[1])  # 记录速度（车身质心）
    v_wheel_log.append(data_in.qvel[adr_qvel_y])  # 记录速度（轮子接触点）
    phi_log.append(x[2])  # 记录 roll
    w_log.append(x[3])  # 记录 roll 角速度
    u_log.append(0.5 * u)  # 记录控制量
    y_ref_log.append(y_ref)  # 记录y参考
    ydot_ref_log.append(ydot_ref)  # 记录y速度参考
    e_y_log.append(e[0])  # 记录y误差
    e_ydot_log.append(e[1])  # 记录y_dot误差
    e_phi_log.append(e[2])  # 记录phi误差
    e_phidot_log.append(e[3])  # 记录phi_dot误差
    u_y_log.append(-K_current[0] * e[0])  # u由y误差产生的分量
    u_ydot_log.append(-K_current[1] * e[1])  # u由y_dot误差产生的分量
    u_phi_log.append(-K_current[2] * e[2])  # u由phi误差产生的分量
    u_phidot_log.append(-K_current[3] * e[3])  # u由phi_dot误差产生的分量


def init_controller():  # 初始化控制器
    global K, x0, x0_joint, qpos0, qvel0, Q, R  # 使用全局变量
    while data.time < settle_time:  # 初始稳定
        data.ctrl[aid_left] = 0.0  # 左轮无力矩
        data.ctrl[aid_right] = 0.0  # 右轮无力矩
        mj.mj_step(model, data)  # 仿真步进
    qpos0 = data.qpos.copy()  # 保存 qpos
    qvel0 = data.qvel.copy()  # 保存 qvel
    x0 = get_state_from_data(data)  # 平衡点状态（关节 y/roll）
    x0[2] = 0.0  # 初始 roll 角
    print(x0)
    x0_joint = get_joint_state_for_linearize(data)  # 平衡点关节状态 看起来没用到
    u0 = np.array([0.0], dtype=float)  # 平衡输入
    Ad, Bd = linearize_discrete_central(x0, u0, qpos0, qvel0)  # 线性化（关节扰动）
    Q = np.diag([5000.0, 200000.0, 3000.0, 8000.0])  # 增大phi和phi_dot权重保持平衡
    R = np.diag([15])  # 适中的R
    # K, S, E = control.dlqr(Ad, Bd, Q, R)  # 离散 LQR
    # 改用 scipy 的底层求解器:
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R) 
    # 根据 DARE 结果手动计算 K 矩阵: 
    K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)  # 计算 LQR 增益
    print("x0 =", x0)  # 输出平衡点
    print("LQR K =", K)  # 输出增益


def controller(model_in, data_in):  # 控制回调
    x = get_state_from_data(data_in)  # 读取状态
    if data_in.time < move_start:  # 先平衡
        y_ref = x0[0]  # y 参考（保持初始）
        ydot_ref = 0.0  # y 速度参考
    elif data_in.time <= move_end:  # 再向 y 方向匀速运动
        t_run = data_in.time - move_start  # 运动阶段时间
        y_ref = x0[0] + vy_cmd * t_run  # 位置参考
        ydot_ref = vy_cmd  # 速度参考
    else:
        y_ref = x0[0] + vy_cmd * (move_end - move_start)  # y 参考
        ydot_ref = 0.0  # y 速度参考
    roll_ref = 0.0  # roll 参考（直立）
    roll_dot_ref = 0.0  # roll 角速度参考
    x_ref = np.array([y_ref, ydot_ref, roll_ref, roll_dot_ref], dtype=float)  # 参考状态
    e = x - x_ref  # 跟踪误差
    u = -K @ e  # LQR 控制
    u = np.clip(u, -16.0, 16.0)  # 力矩限幅
    # print("K =", K)
    # print("e =", e)
    data_in.ctrl[aid_left] = 0.5 * u[0]  # 左轮力矩
    data_in.ctrl[aid_right] = 0.5 * u[0]  # 右轮力矩
    t_log.append(data_in.time)  # 记录时间
    x_log.append(x[0])  # 记录状态
    v_log.append(x[1])  # 记录速度（车身质心）
    v_wheel_log.append(data_in.qvel[adr_qvel_y])  # 记录速度（轮子接触点）
    phi_log.append(x[2])  # 记录 roll
    w_log.append(x[3])  # 记录 roll 角速度
    u_log.append(0.5 * u[0])  # 记录控制量
    y_ref_log.append(y_ref)  # 记录y参考
    ydot_ref_log.append(ydot_ref)  # 记录y速度参考
    e_y_log.append(e[0])  # 记录y误差
    e_ydot_log.append(e[1])  # 记录y_dot误差
    e_phi_log.append(e[2])  # 记录phi误差
    e_phidot_log.append(e[3])  # 记录phi_dot误差
    u_y_log.append(-K[0, 0] * e[0])  # u由y误差产生的分量
    u_ydot_log.append(-K[0, 1] * e[1])  # u由y_dot误差产生的分量
    u_phi_log.append(-K[0, 2] * e[2])  # u由phi误差产生的分量
    u_phidot_log.append(-K[0, 3] * e[3])  # u由phi_dot误差产生的分量


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
    init_LPV_controller()  # 初始化控制器
    data.qpos[adr_qpos_roll] = data.qpos[adr_qpos_roll] + phi_init  # 初始侧倾
    mj.mj_forward(model, data)  # 前向更新
    mj.set_mjcb_control(LPV_controller)  # 设置控制回调

    real_time_start = glfw.get_time()
    sim_time_start = data.time

    while not glfw.window_should_close(window):  # 主循环
        # 1. 计算当前现实世界经过的时间，并乘以速度系数，得到"目标仿真时间"
        real_time_elapsed = glfw.get_time() - real_time_start  # 真实时间
        target_sim_time = sim_time_start + real_time_elapsed * simspeed
        # 2. 让物理引擎快速追赶目标时间 # 如果物理引擎走得慢于现实时间（或者要渲染），它就会在这里连续步进多次
        while data.time < target_sim_time:
            mj.mj_step(model, data)  # 仿真步进
            if data.time >= simend:  # 仿真结束
                break
        if data.time >= simend:  # 仿真结束，退出循环
            glfw.set_window_should_close(window, True)  # 设置窗口关闭标志
            break
        # time_prev = data.time  # 上一时刻
        # while data.time - time_prev < 1.0 / 1000.0:  # 控制渲染频率
        #     mj.mj_step(model, data)  # 仿真步进
        # if data.time >= simend:  # 结束条件
        #     break  # 退出循环
        # 3. 物理引擎追赶完毕，渲染当前状态到屏幕
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
        fig = plt.figure(figsize=(14, 10))  # 创建图

        ax1 = fig.add_subplot(4, 3, 1)  # 子图1
        ax1.plot(t_log, x_log, label="y")  # 画 x
        ax1.plot(t_log, y_ref_log, '--', label="y_ref")  # 画参考
        ax1.set_ylabel(label_x_y)  # y 标签
        ax1.grid(True)  # 网格
        ax1.legend()  # 图例

        ax2 = fig.add_subplot(4, 3, 2)  # 子图2
        ax2.plot(t_log, v_log, label="v_cm")  # 画车身质心速度
        # ax2.plot(t_log, v_wheel_log, label="v_wheel")  # 画轮子速度
        ax2.plot(t_log, ydot_ref_log, '--', label="v_ref")  # 画参考
        ax2.set_ylabel(label_v_y)  # y 标签
        ax2.grid(True)  # 网格
        ax2.legend()  # 图例
        
        ax3 = fig.add_subplot(4, 3, 3)  # 子图3
        ax3.plot(t_log, u_log, label=label_u)  # 画 u
        ax3.set_ylabel(label_u_y)  # y 标签
        ax3.grid(True)  # 网格
        ax3.legend()  # 图例

        ax4 = fig.add_subplot(4, 3, 4)  # 子图4
        ax4.plot(t_log, phi_log, label=label_phi)  # 画 roll
        ax4.set_ylabel(label_phi_y)  # y 标签
        ax4.grid(True)  # 网格
        ax4.legend()  # 图例

        ax5 = fig.add_subplot(4, 3, 5)  # 子图5
        ax5.plot(t_log, w_log, label=label_w_y)  # 画 w
        ax5.set_ylabel(label_w_y)  # y 标签
        ax5.grid(True)  # 网格
        ax5.legend()  # 图例

        # 误差曲线
        ax6 = fig.add_subplot(4, 3, 7)  # 子图6
        ax6.plot(t_log, e_y_log, label="e_y")  # 画y误差
        ax6.set_ylabel("e_y (m)")  # y 标签
        ax6.set_xlabel(label_time)  # x 标签
        ax6.grid(True)  # 网格
        ax6.legend()  # 图例

        ax7 = fig.add_subplot(4, 3, 8)  # 子图7
        ax7.plot(t_log, e_ydot_log, label="e_ydot")  # 画y_dot误差
        ax7.set_ylabel("e_ydot (m/s)")  # y 标签
        ax7.set_xlabel(label_time)  # x 标签
        ax7.grid(True)  # 网格
        ax7.legend()  # 图例

        ax8 = fig.add_subplot(4, 3, 9)  # 子图8
        ax8.plot(t_log, e_phi_log, label="e_phi")  # 画phi误差
        ax8.set_ylabel("e_phi (rad)")  # y 标签
        ax8.set_xlabel(label_time)  # x 标签
        ax8.grid(True)  # 网格
        ax8.legend()  # 图例

        # u分量曲线
        ax9 = fig.add_subplot(4, 3, 10)  # 子图9
        ax9.plot(t_log, u_y_log, label="u_y")  # 画u_y分量
        ax9.plot(t_log, u_ydot_log, label="u_ydot")  # 画u_ydot分量
        ax9.set_ylabel("u分量 (N·m)")  # y 标签
        ax9.set_xlabel(label_time)  # x 标签
        ax9.grid(True)  # 网格
        ax9.legend()  # 图例

        ax10 = fig.add_subplot(4, 3, 11)  # 子图10
        ax10.plot(t_log, u_phi_log, label="u_phi")  # 画u_phi分量
        ax10.plot(t_log, u_phidot_log, label="u_phidot")  # 画u_phidot分量
        ax10.set_ylabel("u分量 (N·m)")  # y 标签
        ax10.set_xlabel(label_time)  # x 标签
        ax10.grid(True)  # 网格
        ax10.legend()  # 图例
        
        out_png = os.path.join(os.path.dirname(__file__), "lqr_roll2d_plot.png")  # 输出路径
        plt.tight_layout()  # 紧凑布局
        plt.savefig(out_png, dpi=200)  # 保存图片
        plt.close(fig)  # 关闭图
        
        # 输出数据到文件方便分析
        out_data = os.path.join(os.path.dirname(__file__), "lqr_roll2d_data.npz")
        np.savez(out_data, t=np.array(t_log), x=np.array(x_log), v=np.array(v_log), 
                 v_wheel=np.array(v_wheel_log),
                 phi=np.array(phi_log), w=np.array(w_log), u=np.array(u_log),
                 y_ref=np.array(y_ref_log), ydot_ref=np.array(ydot_ref_log),
                 e_y=np.array(e_y_log), e_ydot=np.array(e_ydot_log),
                 e_phi=np.array(e_phi_log), e_phidot=np.array(e_phidot_log),
                 u_y=np.array(u_y_log), u_ydot=np.array(u_ydot_log),
                 u_phi=np.array(u_phi_log), u_phidot=np.array(u_phidot_log))
        print(f"数据已保存到: {out_data}")
        print(f"Q = {Q}")
        print(f"R = {R}")
        print(f"K = {K}")
        print(f"roll角最大值: {np.max(np.abs(phi_log)):.4f} rad, roll角最终值: {phi_log[-1]:.4f} rad")
        print(f"roll角速度最大值: {np.max(np.abs(w_log)):.4f} rad/s")
        print(f"控制输入最大值: {np.max(np.abs(u_log)):.4f} N·m")
        print(f"y位置误差最大值: {np.max(np.abs(e_y_log)):.4f} m")
        print(f"y速度误差最大值: {np.max(np.abs(e_ydot_log)):.4f} m/s")


if __name__ == "__main__":  # 入口判断
    main()  # 运行主程序
