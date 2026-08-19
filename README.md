# MuJoCo Sim

`mujoco_sim` 用于把训练得到的 ONNX 运动策略部署到 MuJoCo，并通过 Xbox 类手柄实时控制机器人。当前包含 M20、ZB02W 两类轮足机器人，以及 WH044X 四轮转向底盘；M20 和 ZB02W 均提供平地、粗糙地形入口，ZB02W 还提供 student 策略和深度相机入口。

仿真启动时会把机器人模型、弹丸模型以及可选地形合并成 `tmp_merged.xml`，随后以 MuJoCo 仿真步长运行。控制器每隔 `control_decimation` 个仿真步更新一次；当前入口的控制周期均为 20 ms（50 Hz）。

## 支持的入口

| `--filename` | 机器人 / 场景 | 配置类 | 控制方式 |
|---|---|---|---|
| `m20_flat` | M20 / 平地 | `M20FlatConfig` | RNN ONNX 策略；腿位置 PD + 轮速度 PD |
| `m20_rough` | M20 / 台阶地形 | `M20RoughConfig` | RNN ONNX 策略；额外合并 `rough_stairs.xml` |
| `zb02w_flat` | ZB02W / 平地 | `Zb02wFlatConfig` | RNN ONNX 策略；腿位置 PD + 轮速度 PD |
| `zb02w_rough` | ZB02W / 台阶地形 | `Zb02wRoughConfig` | RNN ONNX 策略；额外合并 `rough_stairs.xml` |
| `zb02w_ts` | ZB02W / 台阶地形 | `Zb02wTsConfig` | student RNN ONNX 策略；启用轮子静止 PID |
| `zb02w_depth` | ZB02W / 台阶地形 | `Zb02wDepthConfig` | TS 控制逻辑；固定式 64×36 深度相机和 OpenCV 预览 |
| `wh044x` | WH044X / 平地 | `Wh044xConfig` | C++ `chassis` 运动学解算；直接写入转向和轮速 actuator control |

## 工作区与外部资源

配置类使用 `pathlib.Path` 根据 `mujoco_sim` 和项目根目录直接构造绝对路径。默认配置依赖如下工作区布局：

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
- WH044X 还要求 `Wh044xConfig` 中的 `xml_path` 和 `chassis_build_dir` 指向有效资源。

如果本地目录不同，请继承对应配置类并覆盖 `policy_path`、`xml_path`、`terrain_xml_path` 或 `chassis_build_dir`。机器人 XML 所在目录必须可写，因为启动时会在该目录生成临时的 `tmp_merged.xml`。

## 安装

建议使用 Python 3.10，并在 `wl_lab` 根目录执行：

```bash
conda create -n ms python=3.10 -y
conda activate ms
pip install -r mujoco_sim/requirements.txt
```

基础依赖包括 `mujoco>=3.9.0`、`numpy`、`onnxruntime`、`pygame` 和 `opencv-python`。MuJoCo 3.9 提供左上角 Base 状态 HUD 和深度离屏渲染使用的接口。此外还需要准备：

- PlotJuggler 遥测（可选）：需要发送调试数据时安装 [`data_vis`](https://github.com/ceaoi/data_vis)。未开启遥测时不会导入该包；开启但包或 UDP socket 不可用时会告警并自动关闭遥测，不影响仿真继续运行。
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
python mujoco_sim/run_script.py --filename=zb02w_ts
python mujoco_sim/run_script.py --filename=zb02w_depth

# WH044X（需要外部模型和 chassis 扩展）
python mujoco_sim/run_script.py --filename=wh044x
```

`--filename` 可以省略 `.py`。入口只会从 `mujoco_sim/scripts/` 中查找脚本，不接受任意文件路径。

### 运行前检查

1. 确认所选配置类中的模型、策略和地形路径存在。
2. 连接索引为 0 的手柄；未连接时仿真仍会尝试运行，但速度指令保持为零并持续输出警告。
3. 如手柄映射不正确，先运行下面的诊断命令查看轴和按键：

```bash
python -m mujoco_sim.utils.gamepad_pygame --index 0 --hz 20
```

## Base 状态 HUD

所有继承 `MujocoDeploy` 的入口都会在 MuJoCo 界面左上角显示 Base 状态，无需增加额外配置。默认 Base 是合并模型中第一个 `<freejoint/>` 所属的 body；当前 M20、ZB02W 和 WH044X 均先合并机器人、再合并弹丸，因此会选择机器人 Base，而不是弹丸的 free joint。

HUD 以固定 20 Hz 刷新，数值保留三位小数；相机和 viewer 画面同步仍保持原有频率。当前 `simulation_dt=0.0005 s` 时，每 100 个仿真步更新一次 HUD。显示字段如下：

| 字段 | 坐标系 | 单位 | 含义 |
|---|---|---|---|
| `z_world` | 世界系 | m | Base 原点的 Z 坐标 |
| `yaw_world` | 世界系 | rad | Base 按 ZYX 欧拉角定义的 yaw，范围为 `[-π, π]` |
| `omega_base` | Base 系 | rad/s | Base 三维角速度 `[x, y, z]`；MuJoCo free-joint 原生角速度 |
| `velocity_base` | Base 系 | m/s | Base 三维线速度 `[x, y, z]`；由 free-joint 世界系线速度旋转到 Base 系 |

界面使用 SI 单位：位置为米、角度为弧度、线速度为米每秒、角速度为弧度每秒。模型必须包含至少一个 free joint；如果没有，初始化会明确报错，因为无法按默认规则确定浮动 Base。

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

- `MujocoDeploy`：接收 `MujocoSimConfig` 配置对象，统一管理可选 PlotJuggler 发送器，并负责合并 XML、创建 MuJoCo model/data、处理手柄、相机、重置、弹丸和实时仿真循环。
- `MujocoDeployWl`：使用 ONNX Runtime 的 CPU provider 推理，支持普通策略和带 `h_in/c_in` 的 RNN 策略，并把 16 维动作拆分为 12 个腿关节位置目标和 4 个轮关节速度目标。
- `MujocoDeployWh`：轻量扩展点；WH044X 的底盘模式切换和运动学控制实现在 `scripts/wh044x.py`。

## PlotJuggler 遥测

轮足配置默认设置 `plotjuggler_enabled=True`，使用单个 `PlotJugglerUDP("127.0.0.1", 5005)` 发送 `actions`、`wheel_vel`、`cmd`、`obs`、`targ_pos`、`targ_vel` 和 `tau`。WH044X 与通用配置默认关闭。

不需要遥测或未安装 `data_vis` 时，可显式关闭：

```python
config = M20FlatConfig(plotjuggler_enabled=False)
```

关闭状态不会导入 `data_vis`。如果启用后导入、创建发送器或发送数据失败，`MujocoDeploy` 会发出一次 warning，将运行时开关以及 `self.config.plotjuggler_enabled` 更新为 `False`，随后停止发送但保持控制和仿真运行。

## ZB02W 深度相机

`zb02w_depth` 继承 `zb02w_ts` 的本体观测、student ONNX 推理和停车 PID，同时使用 MuJoCo 固定相机采集深度。当前 student ONNX 只有 53 维本体观测和 RNN 状态输入，因此深度图不会输入策略，只暴露给调试和后续策略接入。

默认相机固定在 `base_link`，相对位置为 `(0.375, 0.0175, 0.10225) m`，以 60 Hz 输出 64×36 深度图；垂直 FOV 为 47.83°，有效范围为 0.05–3.0 m，绝对近裁剪距离为 0.05 m。`depth_camera_quat` 使用训练端的 `wxyz` 姿态约定，其中 `+X` 为前向、`+Z` 为上方；模型初始化时会自动复合固定轴对齐，将 MuJoCo/OpenGL 相机的本地 `-Z` 观察轴对齐到训练端 `+X`。因此配置中的纯 `+Y` pitch 45° 最终会让相机朝机器人前方并向下 45°。

运行时提供两个数组：

- `depth_image_metric`：形状 `(36, 64)`，单位为米并裁剪到 `[0.05, 3.0]`；
- `depth_image`：形状 `(1, 1, 36, 64)`，近处为 1、远处为 0，可直接作为后续深度策略的输入布局。
- `depth_points_world`：形状 `(N, 3)`，将有效深度按 `depth_pointcloud_stride` 下采样并反投影到 MuJoCo 世界坐标系，单位为米。

OpenCV 窗口使用 `COLORMAP_TURBO` 伪彩色显示近远变化，并默认通过最近邻插值将 64×36 预览放大 4 倍至 256×144；颜色映射仍使用完整的 `depth_min`–`depth_max` 范围，不改变深度数据。MuJoCo 3D viewer 默认同时用红色球体显示深度点云。点云采用相机 `+X` 向右、`+Y` 向上、`-Z` 向前的坐标约定，并通过 `points_camera @ R_world_camera.T + camera_position` 转换到世界坐标。无命中或达到 `depth_max` 的像素不会生成点。

可以分别通过 `depth_camera_display=False` 和 `depth_pointcloud_display=False` 关闭二维窗口或三维点云。若 OpenCV HighGUI 不可用，二维窗口会告警并自动关闭，三维点云与深度采集不受影响。深度入口始终需要可用的 MuJoCo OpenGL 渲染后端；仅关闭预览窗口时可使用 EGL 等离屏后端。相机通过 `MjSpec` 动态加入编译模型，不会修改原始 ZB02W MJCF。

## 轮子静止 PID

RL 策略在停车时仍可能输出很小的非零轮速，单靠轮速度 PD 难以使轮子绝对静止。`MujocoDeployWl` 因此提供可选的停车 PID：保留策略原始轮速目标，并在接近静止时为每个轮子叠加一个以实际轮速为反馈的修正量。该修正是**目标轮速修正**，不是直接施加到 actuator 的力矩；修正后的目标仍由原有轮速度 PD 转换为力矩。

PID 仅在以下条件同时满足时激活：

1. `wheel_stop_pid_enabled` 为 `true`；
2. 三维速度指令满足 `norm(cmd) < 1e-3`；
3. 轮组实际转速满足 `abs(mean(wheel_velocity)) < 2.0`。

对每个轮子，目标静止速度为 0，因此误差为 `error = -wheel_velocity`。控制器使用 `ctrl_dt = simulation_dt * control_decimation` 计算积分项和微分项，并按以下方式修正策略目标：

```text
pid_output = clip(kp * error + ki * integral(error) + kd * derivative(error),
                  -output_limit, output_limit)
target_wheel_velocity = policy_target_wheel_velocity + pid_output
```

PID 输出带有限幅和抗积分饱和：当输出已经饱和且当前误差会让其进一步饱和时，暂停积分累积。首次激活时会把历史误差初始化为当前误差，避免微分项产生突变；PID 退出或仿真重置时，会清空积分、历史误差和激活状态。速度指令非零或轮组速度尚高时，轮速目标完全由策略输出决定。

当前 `Zb02wTsConfig` 已启用该功能，参数为：

```python
@dataclass(frozen=True, kw_only=True)
class Zb02wTsConfig(Zb02wRoughConfig):
    wheel_stop_pid_enabled: bool = True
    wheel_stop_pid_kp: float = 1.0
    wheel_stop_pid_ki: float = 2.0
    wheel_stop_pid_kd: float = 0.00005
    wheel_stop_pid_output_limit: float = 20.0
```

其余现有 M20 / ZB02W 配置默认关闭停车 PID。调参时建议先限制 `wheel_stop_pid_output_limit`，再依次调整 `kp`、`ki` 和 `kd`，并观察停车阶段的各轮速度及目标轮速，避免振荡或积分累积过快。当前代码中的 `wheel_action_vel_deadzone` 仅被读取，相关死区处理已注释，因此该字段不会使策略轮速自动归零。

## 配置要点

配置位于 `mujoco_sim.configs`，均为冻结的关键字参数 dataclass。公共层级如下：

```text
MujocoSimConfig
├── WheelLeggedConfig
│   ├── M20FlatConfig
│   │   └── M20RoughConfig
│   └── Zb02wFlatConfig
│       └── Zb02wRoughConfig
│           └── Zb02wTsConfig
│               └── Zb02wDepthConfig
└── Wh044xConfig
```

Rough 和 TS 配置继承对应机器人的 Flat 配置，只声明有差异的字段。所有序列参数使用 tuple；需要调整配置时应继续创建子类，而不是修改已有配置实例。

| 字段 | 作用 |
|---|---|
| `simulation_dt` | MuJoCo 仿真步长 |
| `control_decimation` | 每多少个仿真步更新一次策略 / 控制指令 |
| `policy_path` | M20 / ZB02W 的 ONNX 策略路径 |
| `xml_path` | 机器人 MJCF 路径 |
| `terrain_xml_path` | 可选地形 MJCF；rough 配置使用 |
| `plotjuggler_enabled` | 是否启用可选 PlotJuggler UDP 遥测；轮足配置默认开启，其他配置默认关闭 |
| `depth_camera_link`, `depth_camera_pos`, `depth_camera_quat` | 深度相机挂载 link、位置及训练端 `+X` 前向坐标约定下的 `wxyz` 姿态；初始化时自动转换到 MuJoCo 相机轴 |
| `depth_camera_width`, `depth_camera_height`, `depth_camera_fovy` | 深度图分辨率及垂直视场角 |
| `depth_camera_near` | 深度相机绝对近裁剪距离；初始化时按 `model.stat.extent` 换算为 MuJoCo `znear` |
| `depth_camera_update_period`, `depth_min`, `depth_max` | 深度更新周期和有效距离范围 |
| `depth_camera_display` | 是否用 OpenCV 实时显示 `COLORMAP_TURBO` 伪彩色深度图 |
| `depth_camera_display_scale` | OpenCV 预览的最近邻整数放大倍数；默认为 `4` |
| `depth_pointcloud_display` | 是否在 MuJoCo 3D viewer 中显示深度点云 |
| `depth_pointcloud_stride`, `depth_pointcloud_radius` | 点云像素采样间隔和球形点半径；默认分别为 `1` 和 `0.01 m` |
| `num_obs`, `num_obs_hist`, `num_actions` | 策略观测、历史和动作维度 |
| `leg_joint_idx`, `wheel_joint_idx` | MuJoCo state 中的腿 / 轮关节索引 |
| `leg_actions_to_mujoco`, `wheel_actions_to_mujoco` | 策略动作到 MuJoCo actuator 的映射 |
| `kpsPos`, `kdsPos`, `kpsVel`, `kdsVel` | 腿位置 PD 与轮速度 PD 增益 |
| `action_scale_pos`, `action_scale_vel` | 策略动作到目标位置 / 速度的缩放 |
| `cmd_range`, `cmd_deadzone` | 手柄速度范围与死区 |
| `max_command_rate` | M20 / ZB02W 指令变化率上限 |
| `wheel_stop_pid_enabled` | 是否启用接近静止时的轮速 PID 修正；未配置时为 `false` |
| `wheel_stop_pid_kp`, `wheel_stop_pid_ki`, `wheel_stop_pid_kd` | 轮子静止 PID 的比例、积分和微分增益；未配置时为 `0.0` |
| `wheel_stop_pid_output_limit` | 单个轮子的 PID 目标轮速修正绝对值上限，必须大于 0；未配置时为 `5.0` |
| `is_rnn` | 是否按 RNN 接口传入并维护隐藏状态 |

部署时，观测的顺序、缩放、关节排列和动作缩放必须与训练端完全一致；修改这些字段前应同时核对训练配置和导出的 ONNX 输入 / 输出。

## 项目结构

```text
mujoco_sim/
├── run_script.py              # 统一入口，根据 --filename 执行 scripts/ 下的脚本
├── configs/                   # dataclass 配置：公共基类及各机器人 Flat/Rough/TS 子类
├── scripts/
│   ├── base/
│   │   ├── base.py            # 通用仿真循环
│   │   ├── base_wl.py         # 轮足 ONNX 推理与 PD 控制
│   │   └── base_wh.py         # WH 类机器人扩展点
│   ├── m20_flat.py
│   ├── m20_rough.py
│   ├── zb02w_flat.py
│   ├── zb02w_rough.py
│   ├── zb02w_ts.py
│   ├── zb02w_depth.py
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

1. 在 `configs/` 中继承对应机器人的 Flat 配置类，只覆盖新场景需要修改或新增的字段。
2. 在 `scripts/` 新建同名入口脚本，实例化该配置类并传给 `MujocoDeploy`、`MujocoDeployWl` 或 `MujocoDeployWh` 子类。
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
