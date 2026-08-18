# MuJoCo Sim — M20 / WH044X 机器人仿真部署

使用 MuJoCo 物理引擎对机器人进行仿真部署和手柄遥控。

## 运行入口

```bash
python ./run_script.py --filename=m20_flat
python ./run_script.py --filename=wh044x
```

`run_script.py` 解析 `--filename` 参数，在 `scripts/` 目录下找到对应脚本并通过 `runpy.run_path` 执行。

## 项目结构

```
mujoco_sim/
├── base.py               # MujocoDeploy 通用基类：仿真循环、手柄、相机、弹丸
├── base_wl.py            # MujocoDeployWl → 四足轮式：ONNX推理 + 腿位置PD + 轮速度PD
├── base_wh.py            # MujocoDeployWh → 半人形底盘：轻量基类（控制逻辑在叶子脚本）
├── run_script.py          # 入口脚本
├── scripts/               # 部署脚本
│   ├── m20_flat.py        # M20 平地部署，含 GaitPhaseGenerator
│   └── wh044x.py          # WH044X 部署，C++ 底盘运动学解算 + 直接设置 qpos/qvel
├── configs/               # YAML 配置
│   ├── m20_flat.yaml      # M20 RNN 策略, sim_dt=0.0005, decimation=40 → 50Hz
│   ├── m20.yaml           # M20 MLP 策略, sim_dt=0.001, decimation=20 → 50Hz
│   └── wh044x.yaml        # WH044X 配置, xml_path + chassis_build_dir
├── utils/
│   ├── deploy_func.py     # quat_rotate / quat_rotate_inverse / pd_ctrl
│   ├── gait_generator.py  # 步态相位/时钟生成（匹配 IsaacLab GaitStateCommand）
│   ├── projectile.py      # 球形弹丸管理器（测试抗冲击）
│   └── gamepad_pygame.py  # 基于 pygame 的手柄封装
├── robots/                # 机器人模型
└── test/
    └── test_base_frame_mujoco.py  # 四元数/坐标系单元测试
```

## 核心架构

**继承链**：
```
MujocoDeploy (base.py)              ← 通用：仿真循环、手柄、相机、弹丸
├── MujocoDeployWl (base_wl.py)     ← 四足轮式：ONNX推理 + 腿位置PD + 轮速度PD
│    └── M20FlatDeploy (scripts/m20_flat.py)
└── MujocoDeployWh (base_wh.py)     ← 半人形底盘：空基类（控制逻辑在叶子脚本）
     └── Wh044xDeploy (scripts/wh044x.py)  ← C++ 底盘运动学 + 直接 qpos/qvel 控制
```

**`MujocoDeploy` (base.py)** 通用基类：
- `__init__`: 加载 YAML → 构建 MuJoCo 模型（合并 robot XML + ball XML）→ 初始化手柄 → 调用 `_init_control()` hook
- `run()`: 主循环，以 `sim_dt` 步进，每 `control_decimation` 步调用一次控制更新
- `step()`: update_cmd → update_obs → update_model_in → update_action → update_tau → mj_step
- `update_cmd()`: 手柄摇杆 → cmd（含死区）
- 抽象方法：`update_obs()`, `update_model_in()`, `update_action()`, `update_tau()`
- Hook：`_init_control()`, `_reset_control()`

**`MujocoDeployWl` (base_wl.py)** 四足轮式：
- `_init_control()`: 加载 WL 配置（关节映射、增益、default_angles）+ ONNX 策略 + RNN state
- `update_action()`: ONNX 推理 → 分解为位置目标（腿）+ 速度目标（轮）
- `update_tau()`: 腿位置 PD + 轮速度 PD
- 配置字段：`leg_joint_idx`, `wheel_joint_idx`, `kpsPos/kdsPos`, `kpsVel/kdsVel`, `default_angles_leg`

**`Wh044xDeploy` (scripts/wh044x.py)** 半人形底盘：
- 仅保留 8 个底盘关节（4 turn + 4 wheel），上身关节在 URDF 中已固定
- `_init_control()`: 初始化 `turn_joint_idx`, `wheel_joint_idx`, `wheel_radius`, C++ `Chassis` 实例
- `update_action()`: 手柄 cmd → C++ 底盘运动学解算（转向角 + 轮速）→ 转为关节位置/速度目标
- `update_tau()`: 直接 `self.tau[:] = self.action`（action 即 tau）
- 配置字段：`xml_path`, `chassis_build_dir`（C++ 底盘库路径）

## 依赖

`mujoco`, `onnxruntime`, `numpy`, `pyyaml`, `pygame`

M20 额外：`data_vis` (PlotJuggler，调试用)
WH044X 额外：C++ chassis 库（`chassis_build_dir` 指向的 build 目录）

## 添加新部署脚本

1. 在 `configs/` 添加 YAML 配置
2. 在 `scripts/` 创建 `xxx.py`，继承对应基类（`MujocoDeploy` / `MujocoDeployWl` / `MujocoDeployWh`）
3. 根据需求实现 `update_obs()`, `update_model_in()`, `update_action()`, `update_tau()` 等
4. 运行 `python ./run_script.py --filename=xxx`
