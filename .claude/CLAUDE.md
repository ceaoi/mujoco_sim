# MuJoCo Sim — M20 四足轮式机器人部署

使用 MuJoCo 物理引擎 + ONNX Runtime 对 M20 机器人 RL 策略进行仿真部署和手柄遥控。

## 运行入口

```bash
python ./mujoco_sim/run_script.py --filename=m20_flat
```

`run_script.py` 解析 `--filename` 参数，在 `scripts/` 目录下找到对应脚本并通过 `runpy.run_path` 执行。

## 项目结构

```
mujoco_sim/
├── base.py               # MujocoDeploy 基类：MuJoCo仿真循环、ONNX推理、PD控制、手柄遥控
├── run_script.py          # 入口脚本
├── scripts/               # 部署脚本（继承 MujocoDeploy，实现 update_obs / update_model_in）
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
│   └── ball/ball.xml       # 弹丸模型
└── test/
    └── test_base_frame_mujoco.py  # 四元数/坐标系单元测试
```

## 核心架构

**`MujocoDeploy` (base.py)** 是核心基类：
- `__init__`: 加载 YAML → 构建 MuJoCo 模型（合并 robot XML + ball XML）→ 加载 ONNX 策略 → 初始化手柄
- `run()`: 主循环，以 `sim_dt` 步进，每 `control_decimation` 步调用一次策略推理
- `step()`: update_cmd → update_obs → update_model_in → update_action(Policy) → update_tau(PD) → mj_step
- `update_action()`: 执行 ONNX 推理（支持 RNN hidden state），将输出分解为位置目标（腿）× 速度目标（轮）
- `update_tau()`: 腿用位置 PD 控制，轮用速度 PD 控制
- 手柄操作：L2=复位, R2=切换跟随/固定相机, A=发射弹丸

**子类只覆盖两个方法**：
- `update_obs(self)`: 从 `self.data` 提取观测（角速度、重力方向、关节位置/速度、历史action、指令）填入 `self.obs`
- `update_model_in(self)`: 设置 `self.model_in`（通常就是 `self.obs`，历史帧用 `self.obs_hist`）

**YAML 配置关键字段**：
- `{mujoco_workspace_dir}` 占位符在加载时替换为 `mujoco_sim/` 目录
- `leg_joint_idx` / `wheel_joint_idx` 定义从 `qpos[7:]` 的索引重排
- `leg_actions_to_mujoco` / `wheel_actions_to_mujoco` 定义 action 到 MuJoCo joint 的映射
- `kpsPos/kdsPos` 腿位置 PD 增益, `kpsVel/kdsVel` 轮速度 PD 增益
- `num_obs`=53, `num_actions`=16 (12 腿位置 + 4 轮速度), `num_obs_hist`=10

## 依赖

`mujoco`, `onnxruntime`, `numpy`, `pyyaml`, `pygame`, `data_vis` (PlotJuggler，仅用于调试 plot)

## 添加新部署脚本

1. 在 `configs/` 添加 YAML 配置文件
2. 在 `scripts/` 创建 `xxx.py`，继承 `MujocoDeploy`，实现 `update_obs()` 和 `update_model_in()`
3. 运行 `python mujoco_sim/run_script.py --filename=xxx`
