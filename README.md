# MuJoCo Sim

`mujoco_sim` 用于把训练得到的 ONNX 运动策略部署到 MuJoCo，并通过 Xbox 类手柄实时控制机器人。当前包含 M20、ZB02W 两类轮足机器人，以及 WH044X 四轮转向底盘；M20 和 ZB02W 均提供平地、粗糙地形两套入口。

仿真启动时会把机器人模型、弹丸模型以及可选地形合并成 `tmp_merged.xml`，随后以 MuJoCo 仿真步长运行。控制器每隔 `control_decimation` 个仿真步更新一次；当前入口的控制周期均为 20 ms（50 Hz）。

## 支持的入口

| `--filename` | 机器人 / 场景 | 配置 | 控制方式 |
|---|---|---|---|
| `m20_flat` | M20 / 平地 | `configs/m20_flat.yaml` | RNN ONNX 策略；腿位置 PD + 轮速度 PD |
| `m20_rough` | M20 / 台阶地形 | `configs/m20_rough.yaml` | RNN ONNX 策略；额外合并 `rough_stairs.xml` |
| `zb02w_flat` | ZB02W / 平地 | `configs/zb02w_flat.yaml` | RNN ONNX 策略；腿位置 PD + 轮速度 PD |
| `zb02w_rough` | ZB02W / 台阶地形 | `configs/zb02w_rough.yaml` | RNN ONNX 策略；额外合并 `rough_stairs.xml` |
| `wh044x` | WH044X / 平地 | `configs/wh044x.yaml` | C++ `chassis` 运动学解算；直接写入转向和轮速 actuator control |

`configs/m20.yaml` 是未被现有入口引用的旧配置，不应直接替代 `m20_flat.yaml`。

## 工作区与外部资源

配置中的 `{mujoco_workspace_dir}` 会在运行时展开为 `mujoco_sim` 的绝对路径。默认配置依赖如下工作区布局：

```text
wl_lab/
├── mujoco_sim/                         # 本项目
├── logs/rsl_rl/.../policy.onnx         # M20 / ZB02W 导出的策略
├── source/.../zb02w/mjcf/mjcf/zb02w.xml
├── robots/wh044x/mjcf/wh044x_generated.xml  # WH044X，需另行提供
└── build/chassis*.so                   # WH044X C++ Python 扩展，需另行构建
```

- M20 的 MJCF 和 mesh 已包含在 `mujoco_sim/robots/M20_mjcf/`。
- ZB02W 的 MJCF 默认从主工作区的 `source/` 目录读取。
- M20、ZB02W 的 ONNX 策略默认从主工作区的 `logs/rsl_rl/` 读取。
- WH044X 还要求 `configs/wh044x.yaml` 中的 `xml_path` 和 `chassis_build_dir` 指向有效资源。

如果本地目录不同，请在对应 YAML 中修改 `policy_path`、`xml_path`、`terrain_xml_path` 或 `chassis_build_dir`。机器人 XML 所在目录必须可写，因为启动时会在该目录生成临时的 `tmp_merged.xml`。

## 安装

建议使用 Python 3.10，并在 `wl_lab` 根目录执行：

```bash
conda create -n ms python=3.10 -y
conda activate ms
pip install -r mujoco_sim/requirements.txt
```

基础依赖包括 `mujoco`、`numpy`、`pyyaml`、`onnxruntime` 和 `pygame`。此外还需要准备：

- M20 / ZB02W：安装 [`data_vis`](https://github.com/ceaoi/data_vis)；当前轮足基类和入口会直接导入 `PlotJugglerUDP`，即使不打开 PlotJuggler 也必须能导入该 Python 包。
- WH044X：编译可被 Python 导入的 `chassis` 扩展，并更新 `chassis_build_dir`。
- MuJoCo viewer 需要图形显示环境；手柄默认按 Linux 下常见的 Xbox 类控制器映射读取。

## 运行

以下命令均从 `wl_lab` 根目录执行：

```bash
# M20
python mujoco_sim/run_script.py --filename=m20_flat
python mujoco_sim/run_script.py --filename=m20_rough

# ZB02W
python mujoco_sim/run_script.py --filename=zb02w_flat
python mujoco_sim/run_script.py --filename=zb02w_rough

# WH044X（需要外部模型和 chassis 扩展）
python mujoco_sim/run_script.py --filename=wh044x
```

`--filename` 可以省略 `.py`。入口只会从 `mujoco_sim/scripts/` 中查找脚本，不接受任意文件路径。

### 运行前检查

1. 确认所选 YAML 中的模型、策略和地形路径存在。
2. 连接索引为 0 的手柄；未连接时仿真仍会尝试运行，但速度指令保持为零并持续输出警告。
3. 如手柄映射不正确，先运行下面的诊断命令查看轴和按键：

```bash
python -m mujoco_sim.utils.gamepad_pygame --index 0 --hz 20
```

## 手柄操作

| 输入 | 功能 |
|---|---|
| 左摇杆 Y | 前后速度 `vx` |
| 左摇杆 X | 横向速度 `vy` |
| 右摇杆 X | 偏航角速度 `yaw` |
| L2 | 重置仿真和控制器状态 |
| R2 | 切换跟随 / 固定相机 |
| A | 从机器人周围生成一个以 6 m/s 撞向机体的弹丸 |
| R1 | 仅 WH044X：在 3 个 chassis 模式间循环切换 |

实际速度范围由对应配置的 `cmd_range` 决定。M20 和 ZB02W 还会按 `max_command_rate` 对指令变化率限幅。

默认 pygame 映射位于 `utils/gamepad_pygame.py`。不同手柄或操作系统的轴编号可能不同，需要调整 `axis_map` / `button_map`。

## 核心流程

```text
run_script.py
└── scripts/<filename>.py
    └── MujocoDeploy.run()
        ├── 每个仿真步：update_tau() → mj_step()
        └── 每 control_decimation 步：
            手柄 → update_cmd() → update_obs()
                 → update_model_in() → update_action()
```

类的职责如下：

- `MujocoDeploy`：加载 YAML、合并 XML、创建 MuJoCo model/data、处理手柄、相机、重置和弹丸，并维护实时仿真循环。
- `MujocoDeployWl`：使用 ONNX Runtime 的 CPU provider 推理，支持普通策略和带 `h_in/c_in` 的 RNN 策略，并把 16 维动作拆分为 12 个腿关节位置目标和 4 个轮关节速度目标。
- `MujocoDeployWh`：轻量扩展点；WH044X 的底盘模式切换和运动学控制实现在 `scripts/wh044x.py`。

## 配置要点

| 字段 | 作用 |
|---|---|
| `simulation_dt` | MuJoCo 仿真步长 |
| `control_decimation` | 每多少个仿真步更新一次策略 / 控制指令 |
| `policy_path` | M20 / ZB02W 的 ONNX 策略路径 |
| `xml_path` | 机器人 MJCF 路径 |
| `terrain_xml_path` | 可选地形 MJCF；rough 配置使用 |
| `num_obs`, `num_obs_hist`, `num_actions` | 策略观测、历史和动作维度 |
| `leg_joint_idx`, `wheel_joint_idx` | MuJoCo state 中的腿 / 轮关节索引 |
| `leg_actions_to_mujoco`, `wheel_actions_to_mujoco` | 策略动作到 MuJoCo actuator 的映射 |
| `kpsPos`, `kdsPos`, `kpsVel`, `kdsVel` | 腿位置 PD 与轮速度 PD 增益 |
| `action_scale_pos`, `action_scale_vel` | 策略动作到目标位置 / 速度的缩放 |
| `cmd_range`, `cmd_deadzone` | 手柄速度范围与死区 |
| `max_command_rate` | M20 / ZB02W 指令变化率上限 |
| `is_rnn` | 是否按 RNN 接口传入并维护隐藏状态 |

部署时，观测的顺序、缩放、关节排列和动作缩放必须与训练端完全一致；修改这些字段前应同时核对训练配置和导出的 ONNX 输入 / 输出。

## 项目结构

```text
mujoco_sim/
├── run_script.py              # 统一入口，根据 --filename 执行 scripts/ 下的脚本
├── configs/                   # 机器人、策略、场景和控制参数
├── scripts/
│   ├── base/
│   │   ├── base.py            # 通用仿真循环
│   │   ├── base_wl.py         # 轮足 ONNX 推理与 PD 控制
│   │   └── base_wh.py         # WH 类机器人扩展点
│   ├── m20_flat.py
│   ├── m20_rough.py
│   ├── zb02w_flat.py
│   ├── zb02w_rough.py
│   └── wh044x.py
├── robots/
│   ├── M20_mjcf/              # M20 模型与 mesh
│   ├── ball/                  # 抗冲击测试弹丸
│   └── terrains/              # rough 台阶地形
├── utils/
│   ├── deploy_func.py         # 四元数和 PD 工具
│   ├── gait_generator.py      # 步态相位生成器
│   ├── gamepad_pygame.py      # pygame 手柄封装与诊断入口
│   ├── projectile.py          # 弹丸管理器
│   └── urdf2xml.py            # URDF → MJCF 转换工具
└── test/                      # 调试脚本与 notebook（部分依赖旧工程路径）
```

## 添加新机器人或场景

1. 在 `configs/` 新增 YAML，并确保模型 / 策略路径可以通过 `{mujoco_workspace_dir}` 正确展开。
2. 在 `scripts/` 新建同名入口脚本，继承 `MujocoDeploy`、`MujocoDeployWl` 或 `MujocoDeployWh`。
3. 按控制方式实现 `update_obs()`、`update_model_in()`、`update_action()`、`update_tau()`；需要初始化或重置内部状态时覆盖 `_init_control()` / `_reset_control()`。
4. 核对 observation、action、关节和 actuator 的维度及顺序。
5. 使用 `python mujoco_sim/run_script.py --filename=<name>` 启动。

如果只是为现有机器人新增地形，可以在配置中增加 `terrain_xml_path` 并复用对应入口逻辑。

## URDF 转 MJCF

`utils/urdf2xml.py` 使用 MuJoCo 官方 Python API 编译 URDF，并自动处理常见 mesh 路径、浮动基座、视觉 / 碰撞分类、灯光、地面和 motor actuator。

```bash
# 默认输出为输入文件同目录下的同名 .xml
python mujoco_sim/utils/urdf2xml.py \
  --path=/path/to/robot.urdf \
  --base=base_link

# 指定输出并允许覆盖
python mujoco_sim/utils/urdf2xml.py \
  --path=/path/to/robot.urdf \
  --base=base_link \
  --output=/path/to/robot.xml \
  --overwrite
```

常用参数：

| 参数 | 说明 |
|---|---|
| `--path` | 必填，输入 URDF |
| `--base` | 必填，接收 `<freejoint/>` 的 body 名称；不存在时会创建包裹 body |
| `--output` | 输出 XML；默认与 URDF 同目录、同名 |
| `--overwrite` | 覆盖已有输出 |
| `--package-map pkg=/path/to/pkg` | 显式指定 ROS package 路径，可重复使用 |
| `--package-root /path/to/root` | 添加自动发现 package 的根目录，可重复使用 |
| `--no-mesh-fix` | 跳过 mesh 路径重写 |
| `--allow-missing-meshes` | mesh 未解析成功时仍继续交给 MuJoCo 编译 |
| `--keep-temp` | 在输出旁保留解析过 mesh 路径的临时 URDF |
| `--discard-visual` | 使用 MuJoCo 默认行为，丢弃纯视觉 mesh |
| `--no-actuator` | 不自动为标量可动关节添加 motor |
| `--motor-force-limit` | 自动生成 motor 的 `ctrlrange` / `forcerange` 绝对值，默认 99 |

查看完整参数：

```bash
python mujoco_sim/utils/urdf2xml.py --help
```
