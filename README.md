## 安装

```bash
# 使用 conda 创建环境
conda create -n ms python=3.10 -y
conda activate ms

# 安装依赖
pip install -r requirements.txt
```

> **注意**：`data_vis` (PlotJuggler) 为自定义仓库（https://github.com/ceaoi/data_vis.git）
>
> 需单独安装，不影响仿真核心功能运行。

## 运行示例

```bash
python ./mujoco_sim/run_script.py --filename=m20_flat
```
