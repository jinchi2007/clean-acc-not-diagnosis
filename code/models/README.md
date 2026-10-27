# code/models — 8 个轻量 1D 基线

由 model-smith 维护（任务 task-4）。8 个模型全部只用 `torch`，**不使用任何预训练权重**
（不引入 timm / torchvision），因此与"数据划分 + 从零训练"的评测协议一致。

## 用法

```bash
cd <工作区根目录>
source setup/env.sh
```

```python
import torch
from code.models import MODELS, build_model

len(MODELS) == 8
m = build_model("wdcnn", num_classes=10, in_len=1024)   # -> nn.Module
m(torch.randn(4, 1, 1024)).shape                        # -> (4, 10)

# 任意 in_len 都可以，权重结构与输入长度无关（除 vit1d/mlp_mixer1d 的位置编码/token 数）
m2048 = build_model("resnet1d", num_classes=10, in_len=2048)

# 首层输入通道数可参数化（默认 1，向后兼容）
m4 = build_model("resnet1d", num_classes=10, in_len=1024, in_ch=4)

# 宽核前端：Conv1d(1,c,k,s)+BN+ReLU -> backbone（自动算 L'=(in_len-k)//s+1）
m_stem = build_stem_model("resnet1d", num_classes=10, in_len=1024, k=64, s=16, c=8)
m_stem(torch.randn(2, 1, 1024)).shape                   # -> (2, 10)
```

统一约定：forward 输入 `(B, 1, L)`，输出 `(B, num_classes)` 的 logits（不含 softmax）。
所有模型都以**全局平均池化 + 线性层**收口（统一分类头，见 `blocks.py: GAPHead`），
因此参数量不随 `in_len` 变化，只有 `vit1d` / `mlp_mixer1d` 的位置编码或 token 数会随长度轻微变化。

## 自测（验收命令）

```bash
python -m code.models.selftest        # 参数表 + MPS 延迟 + in_ch/stem 断言，写 results/model_selftest.txt
python -m code.models.wdcnn           # 单模型 smoke test（每个模块都有 __main__）
```

> 用 `python -m code.models.<name>` 跑单模块时会看到一条 `<frozen runpy>: ... found in
> sys.modules after import of package` 的 RuntimeWarning。这是 runpy 对"包 `__init__` 已经
> 导入该模块"的提示，**无害**；正式自测请用上面的 selftest，不出现该警告。

## 模型清单（结构特征 + 简化点）

| 名称 | 出处 | 保留的结构特征 | 参数量 (in_len=1024) |
|---|---|---|---:|
| `wdcnn` | Zhang et al., *Sensors* 17(2):425, 2017 | 宽首层大核（kernel 64, stride 16, 16 通道）+ 4 层 3x1 卷积/池化 | 34,410 |
| `ticnn` | Zhang et al., *Mech. Syst. Signal Process.* 100:439–453, 2018 | 宽卷积（kernel 64）+ 每层 BN/ReLU 的骨干 | 83,914 |
| `mcnn` | Jiang et al., *IEEE Trans. Ind. Electron.* 66(4):3196–3207, 2019 | 3 个多尺度并联块（kernel 3/7/15 并联后 concat） | 184,554 |
| `resnet1d` | He et al., *CVPR* 2016（1D 适配） | 残差恒等/1x1 投影捷径，3 stage × 2 basic block | 241,322 |
| `ds_cnn` | Howard et al., arXiv:1704.04861；Zhang et al., arXiv:1711.07128 | depthwise + pointwise 可分离卷积堆叠 | 69,642 |
| `lenet1d` | LeCun et al., *Proc. IEEE* 86(11):2278–2324, 1998 | 两层 5x1 卷积 + 池化的经典 LeNet 堆叠 | 3,018 |
| `vit1d` | Dosovitskiy et al., *ICLR* 2021（1D 适配） | patch embedding + 可学习位置编码 + pre-norm Transformer | 139,850 |
| `mlp_mixer1d` | Tolstikhin et al., *NeurIPS* 2021（1D 适配） | token-mixing MLP + channel-mixing MLP 交替 | 747,722 |

统一简化点（每个模块 docstring 里都写清了）：

1. 原论文的分类头（Flatten + 全连接）统一换成 **GAP + 线性**，使参数量与输入长度解耦；
2. `vit1d` / `mlp_mixer1d` 只做 1D 适配并缩到小配置，**没有** 大规模预训练（保持公平）；
3. `ticnn` 的贡献是**训练策略**（干扰/噪声注入），本目录只实现其骨干网络，训练策略属于 `code/train.py`；
4. `lenet1d` 用 GAP 头后容量很小，是刻意的"最弱经典基线"；为进一步对齐 LeNet 原通道数 6/16，
   本实现保持结构、把通道放到 16/32。

所有模型参数量都 < 3M；`selftest` 会断言这一点。

## 输入通道数 `in_ch` 与宽核前端 `build_stem_model`

> 由 task-6 引入，用于回答"多通道可学习前端能否在提升噪声鲁棒性的同时不牺牲干净精度"。

**`in_ch`**：8 个模型的 `__init__` 都新增了尾随关键字参数 `in_ch=1`，首层卷积按它建
（`in_ch` 追加在参数表最后，所以原有位置参数调用完全不受影响，默认行为与 task-4 一致）。
`build_model(name, num_classes, in_len, in_ch=1)` 直接透传。

**`build_stem_model(base_name, num_classes=10, in_len=1024, k=64, s=16, c=1,
activation='relu')`**：在 backbone 前面接一个可学习的前端

```text
x (B,1,L) -> Conv1d(1, c, kernel_size=k, stride=s, bias=False)   # (B, c, L')
          -> BatchNorm1d(c) -> ReLU/GELU                        # (B, c, L')
          -> backbone(in_ch=c)                                  # (B, num_classes)
L' = (in_len - k) // s + 1
```

返回的 `StemModel` 保留 `front` / `base` / `base_name` / `k` / `s` / `c` /
`in_len` / `in_len_stem` 属性，方便日志和消融脚本记录超参。前端把窗口压到 61
（1024, k=64, s=16）或 125（2048）个时间步，backbone 看到的是多通道短序列。
参数量增幅（c=1/4/8）：resnet1d +0.03/+0.39/+0.87%，ds_cnn +0.09/+1.07/+2.37%，
mcnn +0.06/+0.22/+0.47%，vit1d -2.70/-0.36/+2.76%（L' 缩短使位置编码变小，参数量略降），
全部满足"增加 <5%"。

在前端之后 backbone 不再看到原始波形，所以这**不是**数据增广的替代品：它是可学习
降采样 + 通道投影，用 `in_ch>1` 弥补单通道前端（Lead 探针里 c=1、干净精度掉 12–22 点）
的信息损失。

### 消融协议（`stem_ablation.py`）

```bash
python -m code.models.stem_ablation --part 0 --nparts 2   # 两个进程并行（MPS 共享时上限 2）
python -m code.models.stem_ablation --part 1 --nparts 2
python -m code.models.stem_ablation --merge               # 合并 part -> json + md
```

协议与 `code/train.py` 完全一致（split=safe / overlap=0.5 / win=1024 / 15 epoch /
batch=128 / AdamW lr=2e-3 wd=1e-4 / seed=0 / 干净训练），测试端才加高斯噪声
（10/5/0 dB，`code.common.add_noise`，同一 seed）。每个 run 记录参数量、干净精度、
各 SNR 精度和训练耗时，写 `results/stem_ablation.json` 与 `results/stem_ablation.md`。

### 消融结果（13 个配置，1 seed / 15 epoch，干净训练）

协议同上；`Δ` 为相对 **同一 backbone 无前端对照**的百分点差。10 dB 列是 CWRU 干净训练后
直接测噪声的结果，最能区分"结构鲁棒性"。

| config | backbone | stem (k,s,c) | params | clean | 10 dB | 5 dB | 0 dB | Δclean(点) | Δ10dB(点) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| resnet1d_base | resnet1d | none | 241,322 | 1.0000 | 0.4534 | 0.2060 | 0.1000 | 0 | 0 |
| ds_cnn_base | ds_cnn | none | 69,642 | 0.9996 | 0.4224 | 0.2780 | 0.2131 | 0 | 0 |
| vit1d_base | vit1d | none | 139,850 | 0.9963 | 0.7720 | 0.4858 | 0.2127 | 0 | 0 |
| mcnn_base | mcnn | none | 184,554 | 1.0000 | 0.5974 | 0.3097 | 0.1011 | 0 | 0 |
| resnet1d_k64_c1 | resnet1d | (64,16,1) | 241,388 | 0.8728 | 0.7959 | 0.6463 | 0.3690 | **-12.72** | +34.25 |
| resnet1d_k64_c4 | resnet1d | (64,16,4) | 242,258 | 0.9739 | 0.9231 | 0.7907 | 0.4556 | -2.61 | +46.98 |
| resnet1d_k64_c8 | resnet1d | (64,16,8) | 243,418 | 0.9929 | 0.9496 | 0.7910 | 0.3955 | -0.71 | +49.63 |
| resnet1d_k32_c8 | resnet1d | (32,16,8) | 243,162 | 0.9918 | 0.8799 | 0.5619 | 0.3433 | -0.82 | +42.65 |
| resnet1d_k128_c8 | resnet1d | (128,16,8) | 243,930 | 0.9854 | **0.9765** | 0.8799 | 0.4966 | -1.46 | **+52.31** |
| ds_cnn_k64_c4 | ds_cnn | (64,16,4) | 70,386 | 0.9500 | 0.9000 | 0.6623 | 0.2560 | -4.96 | +47.76 |
| ds_cnn_k64_c8 | ds_cnn | (64,16,8) | 71,290 | 0.9799 | 0.9201 | 0.5896 | 0.2224 | -1.98 | +49.78 |
| vit1d_k64_c8 | vit1d | (64,16,8) | 143,706 | 0.9918 | 0.9049 | 0.7474 | 0.3489 | -0.45 | +13.28 |
| mcnn_k64_c8 | mcnn | (64,16,8) | 185,418 | 0.9948 | 0.9884 | **0.9332** | **0.5343** | -0.52 | +39.10 |

**Q：多通道前端能否在提升鲁棒性的同时把干净精度损失压到 ≤3 点？A：能。**
- 通道数是关键旋钮：resnet1d 的 c=1→4→8 对应 Δclean = -12.72 → -2.61 → -0.71 点、
  Δ10dB = +34.25 → +46.98 → +49.63 点。c=1 就是 Lead 早期探针里"信息损失太大"的失败点，
  c=4 已逼近 ≤3 点，c=8 几乎免费（-0.7 点换 +50 点）。
- 核宽 k：c=8 时 k=32/64/128 的 Δclean 都 ≤1.5 点，而 10 dB 单调升 0.880 → 0.950 → 0.977，
  说明"覆盖更长感受野"对噪声最值钱，k=128 是 10 dB 最优、k=64 是甜点。
- 模型族：前端对 resnet1d / ds_cnn 收益最大（+50 点左右），对 vit1d 最小（+13 点，
  因为 vit1d 本身 10 dB 已有 0.772）。
- **最优 (k,s,c)**：按 10 dB 单点 → resnet1d **(128,16,8)**（0.9765，clean -1.46 点）；
  按低 SNR 综合（5/0 dB 全表最好）→ mcnn **(64,16,8)**（0.9332 / 0.5343，clean -0.52 点）。
- 注意口径：1 seed、15 epoch、10 dB 噪声用同一个 seed=0 实现，绝对值有 ±1–2 点抖动；
  但 Δ 量级是 30–50 点，结论方向不受影响。前端是**架构侧**手段，与训练侧的噪声增广
  （Δ10dB 约 +40–50 点、代价 1–9 点）机制不同，二者可叠加。

## 实测：参数量与 MPS 前向延迟

环境：Apple M5 / torch 2.14.1 / MPS / batch=64 / num_classes=10。
延迟为 **20 次连续前向的平均值、取 5 轮的中位数**（小模型单次前向会被调度开销主导，
这样测更稳定，也接近吞吐口径）：

| model | params (1024) | params (2048) | 1024 ms | 2048 ms |
|---|---:|---:|---:|---:|
| wdcnn | 34,410 | 34,410 | 0.42 | 0.55 |
| ticnn | 83,914 | 83,914 | 0.41 | 0.68 |
| mcnn | 184,554 | 184,554 | 3.58 | 7.13 |
| resnet1d | 241,322 | 241,322 | 1.82 | 3.54 |
| ds_cnn | 69,642 | 69,642 | 1.21 | 2.84 |
| lenet1d | 3,018 | 3,018 | 0.58 | 1.21 |
| vit1d | 139,850 | 143,946 | 2.75 | 8.15 |
| mlp_mixer1d | 747,722 | 994,442 | 4.67 | 10.08 |

> 延迟绝对值受机器负载影响很大（MPS 与主网格共享）：同样的 selftest 在不同时间实测
> wdcnn 0.42 / 1.55 ms、resnet1d 1.82 / 4.7–7.7 ms 都出现过（2–3 倍差异）。上表是空载时的
> 参考值，**比较请用同一轮 selftest 里打印的数据**；参数量不受影响。

`python -m code.models.selftest` 还会对每个模型做一次 train 模式前向 + 反向，
确认 MPS 训练通路可用（8/8 通过）。
