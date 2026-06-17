# MuJoCo Sim

基于 MuJoCo 物理引擎的机器人仿真部署项目，支持手柄遥控。

## 支持的机器人

| 机器人 | 脚本 | 说明 |
|---|---|---|
| M20 | `m20_flat` | 四足轮式，ONNX RL 策略推理 |
| WH044X | `wh044x` | 半人形底盘（4 转向 + 4 轮），C++ 运动学解算 |

## 安装

```bash
# 创建 conda 环境
conda create -n ms python=3.10 -y
conda activate ms

# 安装 Python 依赖
pip install -r requirements.txt
```

> `data_vis` (PlotJuggler) 为自定义仓库（https://github.com/ceaoi/data_vis.git），需单独安装，不影响仿真核心功能。

## 运行

```bash
# M20 四足轮式
python ./run_script.py --filename=m20_flat

# WH044X 半人形底盘
python ./run_script.py --filename=wh044x
```

## 手柄操作

| 按键 | 功能 |
|---|---|
| 左摇杆 | 线速度指令 (vx, vy) |
| 右摇杆 X | 角速度指令 (yaw) |
| L2 | 复位 |
| R2 | 切换 跟随/固定 相机 |
| A | 发射弹丸 |

## 项目结构

```
mujoco_sim/
├── base.py              # 通用基类：仿真循环、手柄、相机、弹丸
├── base_wl.py           # 四足轮式基类：ONNX 推理 + PD 控制
├── base_wh.py           # 半人形底盘基类
├── run_script.py        # 入口
├── scripts/             # 部署脚本
├── configs/             # YAML 配置
├── utils/               # 工具（四元数、PD、步态、手柄、弹丸）
├── robots/              # 机器人模型（M20 MJCF）
└── test/                # 单元测试
```

## 添加新机器人

1. 在 `configs/` 添加 YAML 配置
2. 在 `scripts/` 创建部署脚本，继承对应基类
3. 实现 `update_obs()`, `update_model_in()`, `update_action()`, `update_tau()`
4. 运行 `python ./run_script.py --filename=xxx`

# 其他

## URDF 转换工具

`utils/urdf2xml.py` 使用 MuJoCo 官方 API 将 URDF 模型转为 MJCF XML，并自动注入光源、地板、默认类、作动器等仿真所需元素。

```bash
# 基本用法（输出默认在同目录，.urdf → .xml）
python utils/urdf2xml.py --path=/path/to/robot.urdf --base=base_link

# 指定输出路径 + 覆盖已有文件
python utils/urdf2xml.py --path=/path/to/robot.urdf --base=base_link --output=/path/to/robot.xml --overwrite
```

### 命令行参数

| 参数 | 说明 |
|---|---|
| `--path` (必填) | 输入 .urdf 文件路径 |
| `--base` (必填) | 接收 `<freejoint/>` 的 body 名称，如 `--base=base_link`。若该 body 不存在，脚本会自动创建并包裹 `<worldbody>` 子元素 |
| `--output` | 输出 .xml 路径，默认与 URDF 同目录同名 |
| `--overwrite` | 覆盖已有输出文件 |
| `--package-map` | 手动指定包路径映射，如 `--package-map pkg=/path/to/pkg`，可多次使用 |
| `--package-root` | ROS 包文件夹目录，用于自动发现包，可多次使用 |
| `--no-mesh-fix` | 不重写 mesh 路径（跳过绝对路径替换） |
| `--allow-missing-meshes` | 即使部分 mesh 文件找不到也继续编译 |
| `--keep-temp` | 在输出 XML 旁边保存一份解析了 mesh 路径的临时 URDF |
| `--discard-visual` | 恢复 MuJoCo 默认行为，丢弃纯视觉网格（默认会保留） |
| `--no-actuator` | 不自动添加 motor 作动器 |
| `--motor-force-limit` | motor 作动器 ctrlrange/forcerange 限幅，默认 99.0 |

### 后处理流水线

脚本在 URDF → MJCF 编译完成后，自动执行以下后处理：

1. **Robot 默认类层次**：写入 `robot / motor / visual / collision` 四级 default class 树
2. **Geom 分类**：视觉 mesh geom 分配 `group=2` + 无碰撞，碰撞 geom 分配 `group=1` + 完整接触参数
3. **浮动基座**：在 `--base` 指定的 body 上插入 `<freejoint/>`
4. **地板**：添加 checker 纹理 + MatPlane 材质 + 地面平面 geom (`group=0`)
5. **光源**：添加非阴影方向光 `main_light`
6. **Motor 作动器**：为所有标量可动关节添加 `<motor>`（ctrlrange/forcerange = ±99）

---

> 本项目由 Claude Code 协助开发
