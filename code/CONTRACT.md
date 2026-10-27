# 代码契约（先读这个，再写代码）

工作区根目录 = 本文件所在目录的上一级。所有路径都相对它。

## 目录与归属（不要越界写文件）

| 目录 | 负责人 | 内容 |
|---|---|---|
| \`code/common.py\` | lead | 公共工具，**已写好，不要改** |
| \`code/data/\` | data-engineer | 数据下载、解析、滑窗、划分 |
| \`code/models/\` | model-smith | 8 个轻量基线 |
| \`code/train.py\`, \`code/eval/\`, \`code/fab/\`, \`code/analysis/\`, \`experiments/\` | lead | 训练/评测/FAB/分析 |
| \`data/\`, \`results/\`, \`figures/\` | 共享产物 | 缓存与结果 |

## 1. 数据接口（code/data/cwru.py）

\`\`\`python
from code.data.cwru import build_cwru

ds = build_cwru(
    split="leaky",        # "leaky" | "safe" | "crossload"
    overlap=0.5,          # 0.0 <= overlap < 1.0，滑窗重叠率
    win=1024,             # 窗长
    seed=0,               # 只影响随机划分
    task="10class",       # "10class" | "4class"
    root=None,            # 默认 data/raw；缓存写 data/processed
)
# ds 是 dict，必须恰好包含：
#   "x_train": np.float32 (N, 1, win)   已 z-score（逐窗）
#   "y_train": np.int64   (N,)
#   "x_test", "y_test": 同上
#   "meta": {
#       "dataset": "cwru", "split": str, "overlap": float, "win": int,
#       "task": str, "classes": [str, ...],
#       "n_train": int, "n_test": int,
#       "train_groups": [str, ...], "test_groups": [str, ...],   # 源文件 id
#       "seed": int,
#   }
\`\`\`

三种 split 的精确定义（这是论文的核心，必须严格实现）：

- **\`leaky\`**：所有窗先汇成一个大池，再按类别分层随机分 train/test（8:2）。重叠窗会同时出现在两侧 → 泄漏。
- **\`safe\`**：每个源文件内部按时间顺序切块：前 70% 的窗进 train、后 30% 进 test，两块之间留 \`win//2\` 个采样点的**隔离带**（隔离带内的窗丢弃），保证任何 train 窗与 test 窗不共享采样点。
- **\`crossload\`**：按负载划分，train = 负载 {0,1} hp，test = 负载 {2,3} hp（与 overlap 无关，按窗切，全部来源）。

缓存：\`data/processed/cwru_{task}_{split}_ov{overlap}_{win}.npz\`。重复调用必须命中缓存（毫秒级）。

## 2. 模型接口（code/models/__init__.py）

\`\`\`python
from code.models import MODELS, build_model

len(MODELS) == 8
m = build_model("wdcnn", num_classes=10, in_len=1024)   # -> nn.Module
# m(torch.randn(4, 1, 1024)) -> (4, 10)
\`\`\`

必须实现的 8 个基线（名称固定，全小写下划线）：
\`wdcnn\`, \`ticnn\`, \`mcnn\`, \`resnet1d\`, \`ds_cnn\`, \`lenet1d\`, \`vit1d\`, \`mlp_mixer1d\`

要求：任意 \`in_len\`（1024/2048）可用；参数量 < 3M；docstring 注明原始论文；只用 torch。

## 3. 编码规范

- Python 3.12，numpy + torch 真跑通；每个模块自带 \`if __name__ == "__main__":\` 的 smoke test。
- 不打印大量日志；训练日志交给 lead 的 train.py。
- 不用 dataclass 之外的第三方库；不写 \`~/Library\`、\`/tmp\` 之外的路径（\`/tmp\` 也别用）。
- 完成后必须自己跑一遍 smoke test，把**真实输出**贴进交付说明，并在 \`code/<你的目录>/README.md\` 里写一句用法。

## 4. 激活环境

\`\`\`bash
cd <工作区根目录>
source setup/env.sh && python -c "import torch; print(torch.backends.mps.is_available())"
\`\`\`
