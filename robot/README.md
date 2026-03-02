# 简化轮腿机器人（MuJoCo）
本目录用于科研级仿真验证：包含完整三维模型、二维平衡模型、以及舵轮解算控制脚本。质量与几何由模型自身定义，重心由 MuJoCo 自动计算。

## 文件结构
- `robot.xml`：完整机器人模型（MJCF）
- `scene.xml`：带地面/光源/相机的加载场景
- `ip2d.xml`：二维约束模型（roll 平衡用）
- `lqr_roll2d.py`：二维平衡 LQR（只控 roll/roll_dot）
- `lqr_roll_move_swerve.py`：平衡 + 舵轮运动（键盘控制）
- `steer_cart.xml`：平面舵轮小车模型（锁死 roll/pitch）
- `steer_cart_control.py`：方向键控制的舵轮小车

## 坐标系约定
- **x**：向前  
- **y**：向左  
- **z**：向上  

## 主要关节（robot.xml / ip2d.xml）
- `base_x`, `base_y`, `base_z`：平动  
- `base_roll`, `base_1`, `base_2`：转动  
- `left_steer`, `right_steer`：舵角（绕 z）  
- `left_wheel_spin`, `right_wheel_spin`：轮子自转（绕局部 y）

## 执行器
- `left_steer_pos`, `right_steer_pos`：舵角位置控制（rad）
- `left_wheel_tau`, `right_wheel_tau`：轮子力矩控制（N·m）

## 快速开始
1. 直接在 MuJoCo 中加载 `scene.xml`  
2. 运行二维平衡：  
   `python lqr_roll2d.py`
3. 运行平衡 + 舵轮运动：  
   `python lqr_roll_move_swerve.py`

## 说明
- `ip2d.xml` 用于 LQR 平衡，要求前后轮方向一致  
- 舵轮运动使用 `lqr_roll_move_swerve.py`，速度由方向键控制  
