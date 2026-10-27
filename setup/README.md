# 工作区深度学习实验环境（Apple M5 / MPS）

在 Mac 上从零搭建的、完全自包含（不写 `~/Library`、`/usr/local`，不用 sudo/brew）的
PyTorch 实验环境。所有解释器、依赖、缓存和 Jupyter kernel 都在本工作区内，可以整目录复制、
删除重建。

| 项目 | 值 |
| --- | --- |
| 机器 | Apple M5，16 GB 统一内存 |
| 系统 | macOS 26.5.1 (Darwin 25F80, arm64) |
| Python | 3.12.16（uv 管理的独立 CPython，`tools/python/`） |
| PyTorch | 2.14.1 + torchvision 0.29.1（macOS arm64 wheel，MPS 后端） |
| MPS | **available=True**（实测可用，1D-CNN 训练比 CPU 快约 5.6x） |
| uv | 0.13.6（独立二进制 `tools/bin/uv`） |

---

## 1. 从零复现（一条命令）

```bash
cd /Users/hanjinchi/Documents/deepseek-harness/default-workspace
bash setup/install.sh
```

`install.sh` 会依次完成（幂等，可重复执行）：

1. 从 GitHub releases 下载 macOS arm64 的 uv 独立二进制 → `tools/bin/uv`（失败时可用
   `pip install uv` 的 wheel 解包替代，本项目未走该退路）。
2. `UV_PYTHON_INSTALL_DIR=tools/python` 安装私有 CPython 3.12（`uv python install 3.12 --no-bin`，
   `--no-bin` 保证不往 `~/.local/bin` 写软链）。
3. `uv venv --python 3.12 .venv` 创建虚拟环境。
4. `uv pip install` 安装全部依赖，**优先清华镜像** `https://pypi.tuna.tsinghua.edu.cn/simple`，
   失败自动回退官方 `https://pypi.org/simple`。
5. `setup/register_kernel.sh` 写 `.jupyter/kernels/dsh-pytorch/kernel.json`（不写 `~/Library`）。
6. 运行 `setup/verify_env.py` 做端到端验证。

首次执行（冷缓存，含 121 MB 的 torch wheel）约 2~3 分钟；热缓存实测 **32 秒**。

## 2. 日常使用

```bash
cd /Users/hanjinchi/Documents/deepseek-harness/default-workspace
source setup/env.sh          # 激活 .venv + 设置全部工作区内缓存/Jupyter 变量

python setup/verify_env.py   # 环境体检：版本 / MPS / 1D-CNN 训练 / resnet18 前向 / 内存
python setup/bench_mps.py    # CPU vs MPS 基准（默认 3 epoch，可加 --epochs --batch-size）

jupyter lab                  # kernel 里选 "Python 3.12 (dsh MPS)"
```

> 注意：`jupyter lab` 必须在执行过 `source setup/env.sh` 的 shell 里启动，
> 否则找不到 `.jupyter/` 里的 kernel（`JUPYTER_PATH` 指过去）。

## 3. verify_env.py 实测输出（一次 from-zero 重装后）

```text
====================================================================
ENVIRONMENT
====================================================================
python           : 3.12.16 (CPython)
executable       : /Users/hanjinchi/Documents/deepseek-harness/default-workspace/.venv/bin/python
platform         : macOS-26.5.1-arm64-arm-64bit
machine          : arm64
torch            : 2.14.1 (git 5c488690)
torch mps built  : True
--------------------------------------------------------------------
torch            : 2.14.1
torchvision      : 0.29.1
numpy            : 2.5.4
pandas           : 3.1.0
scipy            : 1.18.1
scikit-learn     : 1.9.1
matplotlib       : 3.11.2
seaborn          : 0.13.2
jupyterlab       : 4.6.4
ipykernel        : 7.4.0
tqdm             : 4.70.1
PyYAML           : 6.0.3
timm             : 1.0.30
einops           : 0.8.2
torchinfo        : 1.8.0
captum           : 0.9.0
ptflops          : 0.7.5
openpyxl         : 3.1.5
umap-learn       : 0.5.12
--------------------------------------------------------------------
====================================================================
MPS / DEVICE
====================================================================
torch.backends.mps.is_built()     : True
torch.backends.mps.is_available() : True
selected device                   : mps
====================================================================
1D-CNN TRAINING on mps (synthetic bearing vibration, 2048x1x1024, 3 epochs)
====================================================================
  warm-up batch done (first-launch shader compilation excluded)
  epoch 1/3:    198.0 ms  (   10343 samples/s)  loss=0.1752  acc=0.982
  epoch 2/3:    155.4 ms  (   13180 samples/s)  loss=0.0073  acc=1.000
  epoch 3/3:    149.1 ms  (   13737 samples/s)  loss=0.0037  acc=1.000
  mean epoch time : 167.5 ms   final loss: 0.0037
====================================================================
resnet18 FORWARD on mps (batch=32, 3x224x224, weights=None)
====================================================================
  49.7 ms / batch   1.55 ms / image   (643.4 images/s)
====================================================================
MEMORY
====================================================================
process peak RSS (ru_maxrss)   : 542 MB
mps current_allocated_memory   : 0.0 MB
mps driver_allocated_memory    : 1121.9 MB
mps recommended_max_working_set: 11.8 GB
cpu cores                      : 10
====================================================================
RESULT: PASS  (mps available=True)
====================================================================
```

关键结论：**`torch.backends.mps.is_available() == True`，`RESULT: PASS`**；
1D-CNN 在 2048 条 1x1024 合成轴承振动数据上（batch=64）稳定 **~150-200 ms/epoch**
（约 1.1-1.4 万样本/秒），resnet18 前向 32x3x224x224 **~50 ms/batch**（1.55 ms/图）。

## 4. CPU vs MPS 实测（setup/bench_mps.py）

```text
workload: 1D-CNN, 4096 x 1 x 1024, batch=64, epochs=3, threads=4
torch 2.14.1, mps available=True

== CPU ==
    epoch  1:   1650.8 ms       2481 samples/s  loss=0.0934  acc=0.987
    epoch  2:   1665.4 ms       2459 samples/s  loss=0.0033  acc=1.000
    epoch  3:   1675.6 ms       2444 samples/s  loss=0.0017  acc=1.000
== MPS ==
    epoch  1:    321.4 ms      12743 samples/s  loss=0.0934  acc=0.987
    epoch  2:    266.8 ms      15353 samples/s  loss=0.0033  acc=1.000
    epoch  3:    297.5 ms      13769 samples/s  loss=0.0017  acc=1.000

==================================================================
device     mean epoch (ms)    train samples/s   infer samples/s
------------------------------------------------------------------
cpu                 1664.0               2462              3586
mps                  295.2              13874             20541
------------------------------------------------------------------
speed-up on MPS: train 5.64x, infer 5.73x
==================================================================
```

| device | mean epoch | train 吞吐 | infer 吞吐 |
| --- | --- | --- | --- |
| CPU（默认 4 线程） | 1664.0 ms | 2462 samples/s | 3586 samples/s |
| MPS (M5 GPU) | 295.2 ms | 13874 samples/s | 20541 samples/s |
| 加速比 | **5.64x** | 5.6x | 5.7x |

注：小模型 + 小 batch 时 MPS 的优势会被 kernel launch 开销吃掉一部分，模型越大收益越明显。

### 4.1 CWRU 量级训练耗时（给实验排期用）

同样 1D-CNN，12 288 个 1x1024 窗口（≈ CWRU 单负载切窗量级），batch=128，warm-up 后实测：

| device | epoch 耗时 | 吞吐 | 50 epoch 一次实验 |
| --- | --- | --- | --- |
| CPU（4 线程） | ~5.0 s | ~2 460 samples/s | ~4 min |
| MPS | ~0.7 s | ~1.6-1.8 万 samples/s | **~35 s** |

结论：在这台 M5 上，CWRU 量级的单次训练是"几十秒"级，3 个月跑 100+ 组消融实验完全可行。

## 5. Jupyter kernel

kernel spec 在工作区内：`.jupyter/kernels/dsh-pytorch/kernel.json`，显示名
**"Python 3.12 (dsh MPS)"**。验证方式：

```bash
source setup/env.sh
jupyter kernelspec list        # 应看到 dsh-pytorch 指向本工作区
jupyter lab
```

已实测两件事：

1. 用 `jupyter_client` 启动该 kernel 并执行 `import torch` / MPS 检查，返回
   `kernel python ok, mps = True 2.14.1`；
2. `jupyter lab --no-browser --port 8899` 能正常起服务，`/api/kernelspecs` 里能看到
   `dsh-pytorch`（display_name = "Python 3.12 (dsh MPS)"）。

## 6. 目录结构 / 磁盘占用

```text
.
├── .venv/                1.2 GB   # Python 3.12 虚拟环境（依赖都在这）
├── .jupyter/             4 KB    # 工作区内 Jupyter kernel spec
├── setup/                44 KB   # env.sh / install.sh / register_kernel.sh / verify_env.py / bench_mps.py / README
└── tools/                1.3 GB
    ├── bin/uv            uv 0.13.6 独立二进制
    ├── python/           72 MB    # 私有 CPython 3.12.16
    ├── uv-cache/         1.1 GB   # uv 包缓存（可删，删后重装会重新下载）
    ├── downloads/        14 MB    # uv tar 包（离线重装用）
    └── logs/             安装日志（pip-install.log, install-from-zero.log）
```

清理缓存：`rm -rf tools/uv-cache`（省 1.1 GB，不影响已装好的 .venv）。

## 7. setup/env.sh 设置的关键变量

| 变量 | 指向 | 作用 |
| --- | --- | --- |
| `UV_PYTHON_INSTALL_DIR` | `tools/python` | uv 安装的 Python 放在工作区 |
| `UV_CACHE_DIR` | `tools/uv-cache` | 包缓存放工作区 |
| `UV_TOOL_DIR` | `tools/uv-tools` | uv tool 目录放工作区 |
| `UV_PYTHON_BIN_DIR` | `tools/bin` | 不往 `~/.local/bin` 写软链 |
| `HF_HOME` | `tools/hf-cache` | HuggingFace 模型/数据集缓存 |
| `TORCH_HOME` | `tools/torch-cache` | torch.hub 权重缓存（如 timm） |
| `MPLCONFIGDIR` | `tools/mpl-cache` | matplotlib 字体缓存（默认写 `~/.config`） |
| `JUPYTER_PATH` | `.jupyter` | 让 jupyter 找到工作区内 kernel |
| `JUPYTER_DATA_DIR` / `JUPYTER_CONFIG_DIR` / `JUPYTER_RUNTIME_DIR` / `IPYTHONDIR` | `tools/jupyter-*`、`tools/ipython` | 避免写 `~/Library/Jupyter` |

## 8. 已安装的包

核心：`torch 2.14.1`、`torchvision 0.29.1`、`numpy 2.5.4`、`pandas 3.1.0`、`scipy 1.18.1`、
`scikit-learn 1.9.1`、`matplotlib 3.11.2`、`seaborn 0.13.2`、`jupyterlab 4.6.4`、`ipykernel 7.4.0`。

实验/工具链：`tqdm`、`pyyaml`、`timm 1.0.30`、`einops 0.8.2`、`torchinfo 1.8.0`、`captum 0.9.0`、
`ptflops 0.7.5`、`openpyxl 3.1.5`、`umap-learn 0.5.12`（连带 numba 0.68.0 / llvmlite 0.50.0）。
共 134 个包，完整清单见 `tools/logs/pip-install.log`。

## 9. 踩坑记录（复现时会遇到的真实问题）

1. **`uv python install` 默认要在 `~/.local/bin` 建可执行链接**，被沙箱拒绝（`Operation not permitted`）。
   解法：`uv python install --no-bin` + `UV_PYTHON_BIN_DIR=tools/bin`，不碰工作区外目录。
2. **首次 MPS 运算要编译 Metal shader**：第一次 1D-CNN epoch 实测 ~4.0 s，而稳态只要 ~0.15 s。
   `verify_env.py` / `bench_mps.py` 都加了 warm-up batch 并把首轮排除在计时外，否则均值会被
   严重高估（修正前 mean epoch 是 1408 ms，修正后 167 ms）。
3. **缓存默认位置全在工作区外**：matplotlib 字体缓存写 `~/.config/matplotlib`、HF 权重写
   `~/.cache/huggingface`、torch.hub 写 `~/.cache/torch`，沙箱下会报权限错误。解法：全部用环境变量
   重定向到 `tools/` 下（见第 7 节）。
4. **`jupyter kernelspec install --user` 会写 `~/Library/Jupyter`**，被拒。解法：手写
   `.jupyter/kernels/dsh-pytorch/kernel.json`（argv 用 `.venv/bin/python` 绝对路径）+ `JUPYTER_PATH`。
5. **Jupyter 运行时目录**：`JUPYTER_RUNTIME_DIR` 默认在 `~/Library/Jupyter/runtime`，不设置的话
   kernel 启动可能失败，需显式指到工作区。
6. **网络**：清华镜像可用（torch arm64 wheel 121 MB 下载正常），官方源作为回退；uv 二进制走
   GitHub releases 也能访问。系统 python3.9 的退路没有用上（uv 路线一次成功）。
7. **MPS 内存上限**：16 GB 统一内存下 `recommended_max_working_set` 约 11.8 GB；大 batch/大模型时
   注意 OOM。MPS 也**不支持 float64**，遇到未实现算子可用 `PYTORCH_ENABLE_MPS_FALLBACK=1`。
8. **CPU 默认线程数只有 4**（torch 在 M5 上检测到的性能核数），所以 CPU 基线偏低；对比时注意。

## 10. 验收命令（复制即用）

```bash
cd /Users/hanjinchi/Documents/deepseek-harness/default-workspace
source .venv/bin/activate
python setup/verify_env.py && echo "ACCEPTANCE OK"
```

验收标准：退出码 0、输出末尾 `RESULT: PASS (mps available=True)`。