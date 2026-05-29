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

---

> 本项目由 Claude Code 协助开发
