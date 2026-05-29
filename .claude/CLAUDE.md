# MuJoCo Sim — M20 四足轮式机器人部署

使用 MuJoCo 物理引擎对机器人 RL 策略进行仿真部署和手柄遥控。

## 运行入口

```bash
python ./mujoco_sim/run_script.py --filename=m20_flat
```

`run_script.py` 解析 `--filename` 参数，在 `scripts/` 目录下找到对应脚本并通过 `runpy.run_path` 执行。

## 项目结构

```
mujoco_sim/
├── base.py               # MujocoDeploy 通用基类：仿真循环、手柄、相机、弹丸（不含 ONNX）
├── base_wl.py            # MujocoDeployWl(WL) → 四足轮式：ONNX推理 + 腿位置PD + 轮速度PD
├── run_script.py          # 入口脚本
├── scripts/               # 部署脚本（继承对应形态基类，实现 update_obs / update_model_in）
│   └── m20_flat.py        # M20 平地部署，含 GaitGenerator
├── configs/               # YAML 配置（策略路径、关节映射、增益、维度等）
│   ├── m20_flat.yaml      # RNN 策略, sim_dt=0.0005, decimation=40 → 50Hz 控制
│   └── m20.yaml           # MLP 策略, sim_dt=0.001, decimation=20 → 50Hz 控制
├── utils/
│   ├── deploy_func.py     # quat_rotate / quat_rotate_inverse / pd_ctrl
│   ├── gait_generator.py  # 步态相位/时钟生成（匹配 IsaacLab GaitStateCommand）
│   ├── projectile.py      # 球形弹丸管理器（测试抗冲击）
│   └── gamepad_pygame.py  # 基于 pygame 的手柄封装
├── robots/
│   ├── M20_mjcf/           # M20 机器人 MuJoCo XML + STL 网格文件
│   ├── wh044x → ../../robots/wh044x  # 半人形机器人（URDF）
│   └── ball/ball.xml       # 弹丸模型
└── test/
    └── test_base_frame_mujoco.py  # 四元数/坐标系单元测试
```

## 核心架构

**继承链**：
```
MujocoDeploy (base.py)          ← 通用：仿真循环、手柄、相机、弹丸
  └── MujocoDeployWl (base_wl.py)  ← WL形态：ONNX推理 + 腿位置PD + 轮速度PD
       └── M20FlatDeploy (scripts/m20_flat.py)  ← 实现 update_obs / update_model_in
```

**`MujocoDeploy` (base.py)** 通用基类：
- `__init__`: 加载 YAML → 构建 MuJoCo 模型（合并 robot XML + ball XML）→ 初始化手柄 → 调用 `_init_control()` hook
- `run()`: 主循环，以 `sim_dt` 步进，每 `control_decimation` 步调用一次控制更新
- `step()`: update_cmd → update_obs → update_model_in → update_action → update_tau → mj_step
- `update_cmd()`: 手柄摇杆 → cmd（含死区）
- 抽象方法：`update_obs()`, `update_model_in()`, `update_action()`, `update_tau()`
- Hook：`_init_control()`, `_reset_control()`（子类初始化/重置控制专用状态）
- 手柄操作：L2=复位, R2=切换跟随/固定相机, A=发射弹丸

**`MujocoDeployWl` (base_wl.py)** WL形态：
- `_init_control()`: 加载 WL 专用配置（关节映射、增益、default_angles 等）+ ONNX 策略 + RNN state
- `update_action()`: 执行 ONNX 推理 → 分解为位置目标（腿）+ 速度目标（轮）
- `update_tau()`: 腿用位置 PD 控制，轮用速度 PD 控制
- `_reset_control()`: 重置控制目标 + RNN state

**叶子脚本只覆盖两个方法**：
- `update_obs(self)`: 从 `self.data` 提取观测填入 `self.obs`
- `update_model_in(self)`: 设置 `self.model_in`

**YAML 配置关键字段**（WL）：
- `{mujoco_workspace_dir}` 占位符在加载时替换为 `mujoco_sim/` 目录
- `leg_joint_idx` / `wheel_joint_idx` 定义从 `qpos[7:]` 的索引重排
- `leg_actions_to_mujoco` / `wheel_actions_to_mujoco` 定义 action 到 MuJoCo joint 的映射
- `kpsPos/kdsPos` 腿位置 PD 增益, `kpsVel/kdsVel` 轮速度 PD 增益
- `num_obs`=53, `num_actions`=16 (12 腿位置 + 4 轮速度), `num_obs_hist`=10

## 依赖

`mujoco`, `onnxruntime`, `numpy`, `pyyaml`, `pygame`, `data_vis` (PlotJuggler，仅用于调试 plot)

## 添加新机器人

1. 在 `configs/` 添加 YAML 配置文件
2. 在 `scripts/` 创建 `xxx.py`，继承对应形态基类（如 `MujocoDeployWl`），实现 `update_obs()` 和 `update_model_in()`
3. 运行 `python mujoco_sim/run_script.py --filename=xxx`
