---
comments: true
---

# 并行训练（Parallel Training）

!!! tip "核心要点"
    大模型训练 = **显存 + 通信 + 计算**的三角博弈。四种并行策略各有取舍，实际训练混合使用。理解每种策略的**通信量、显存分布、计算效率**是核心。

---

## 为什么需要分布式训练？

### 显存爆炸：一个具体例子

以 GPT-3（175B 参数）为例，算一下单卡要多少显存：

| 组件 | 计算 | 大小 |
|------|------|------|
| FP16 模型参数 | 175B × 2 bytes | **350 GB** |
| FP16 梯度 | 同参数 | **350 GB** |
| FP32 Optimizer States (Adam) | 175B × 3 × 4 bytes (param + m + v) | **2100 GB** |
| FP32 参数副本 (混合精度) | 175B × 4 bytes | **700 GB** |
| **总计** | | **~3500 GB** |

对比：单张 A100 只有 **80 GB** 显存 → 需要至少 **44 张卡**才能装下模型和优化器状态，还不算中间激活。

### 激活显存：被忽视的大头

以 Transformer 为例，一层 Self-Attention 的激活：

```
每个 token 需要存储的激活：
├── Q, K, V: 各 [bsz, seq_len, head_dim] — 3 个 tensor
├── QK^T 矩阵: [bsz, n_heads, seq_len, seq_len] — 最大的一块！
├── Attention weights (softmax后): 同上
├── Attention output: [bsz, seq_len, hidden_dim]
└── MLP 中间激活: [bsz, seq_len, 4×hidden_dim] ← 两次

总激活 ≈ bsz × seq_len × hidden_dim × (34 + 5×n_heads×seq_len/hidden_dim)
```

**例子**：hidden_dim=4096, seq_len=2048, bsz=8, n_heads=32, 80 层：
```
单层激活 ≈ 8 × 2048 × 4096 × 50 × 2 bytes ≈ 6.7 GB
80 层 → 如果不做重计算，激活需要 536 GB！
```

这就是为什么 **Activation Checkpointing（激活重计算）** 几乎是必选项。

---

## 并行策略全景

```
                      ┌───────────────┐
                      │  Data Parallel │  ← 数据切分，模型完整副本
                      └───────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              │       Model Parallel          │
              └───────────────┬───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  ┌─────┴─────┐      ┌───────┴────────┐    ┌───────┴────────┐
  │  Tensor   │      │   Pipeline      │    │   Sequence     │
  │  Parallel │      │   Parallel      │    │   Parallel     │
  │ (层内切分) │      │  (层间切分)      │    │ (序列维度切分)  │
  └───────────┘      └────────────────┘    └────────────────┘
```

**三者正交，可以组合！** 实际大模型训练 = DP × TP × PP 的三维并行网格。

---

## 1. Data Parallel（数据并行）

### 1.1 朴素 DP vs DDP

**朴素 DP（Parameter Server）**：

```
Worker 0 ──→ gradients ──→ [Parameter Server] ──→ updated weights ──→ Worker 0
Worker 1 ──→ gradients ──→                           ──→ Worker 1
Worker 2 ──→ gradients ──→                           ──→ Worker 2

问题：PS 成为通信瓶颈（单点聚合），所有梯度集中到一个节点
```

**DDP（Distributed Data Parallel）**：

```
Worker 0 ←── AllReduce gradients ──→ Worker 1
    ↕                                    ↕
Worker 2 ←── AllReduce gradients ──→ Worker 3

每个 Worker 直接参与 AllReduce，无中心瓶颈
PyTorch DDP 默认使用 Ring AllReduce
```

### 1.2 DDP 训练一步的完整流程

```
Step N 在不同 GPU 上的时间线：

GPU 0:  [FWD────────][BWD────────][AllReduce──][Optimizer Step]
GPU 1:  [FWD────────][BWD────────][AllReduce──][Optimizer Step]
GPU 2:  [FWD────────][BWD────────][AllReduce──][Optimizer Step]

每个 GPU 拿到不同的 mini-batch 切片：
  global_batch_size = 256
  GPU 数 = 8
  → 每个 GPU 的 per_gpu_batch_size = 256/8 = 32
```

### 1.3 通信与计算重叠（Overlap）

PyTorch DDP 的核心优化——**梯度 AllReduce 可以和反向传播重叠**：

```
Layer 4: [BWD][AllReduce grad_L4──]────────────────
Layer 3:       [BWD][AllReduce grad_L3──]──────────
Layer 2:              [BWD][AllReduce grad_L2──]───
Layer 1:                     [BWD][AllReduce grad_L1]

→ 不等所有梯度算完再通信，而是"算完一层就传一层"
→ DDP 将参数注册到"桶（bucket）"中，桶满就触发 AllReduce
```

实际代码（PyTorch）：

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化
dist.init_process_group(backend=''nccl'')
model = DDP(model, device_ids=[local_rank])

# 训练循环（和单卡几乎一样！）
for data, label in dataloader:
    loss = model(data)          # Forward（各 GPU 独立）
    loss.backward()             # Backward（边算边 AllReduce 梯度）
    optimizer.step()            # 所有 GPU 权重一致
```

### 1.4 梯度累积（Gradient Accumulation）—— 用小显存模拟大 Batch

```python
# 想跑 global_batch_size=256 但单卡只能放下 batch_size=32
# 解法：累积 8 步再做一次同步

accumulation_steps = 8
for i, (data, label) in enumerate(dataloader):
    loss = model(data) / accumulation_steps  # 注意除以累积步数
    loss.backward()                          # 梯度累加在 .grad 中
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()    # 此时梯度 = 8 个 micro-batch 的平均
        optimizer.zero_grad()
```

**警告**：Batch Normalization 在累积模式下，running statistics 是基于 micro-batch 而非 global batch 的，会导致统计量偏差。

### 1.5 Sync BatchNorm —— 跨 GPU 同步 BN 统计量

普通 BN 在 DDP 下，每个 GPU 独立计算 mini-batch 的 μ 和 σ² → 统计不准。

```python
# PyTorch 内置方案
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
# 现在 BN 层会 AllReduce 通信来同步 μ 和 σ²

# 通信量：每个 BN 层 2×num_features 个 float
# 对于 ResNet-50 (约 50 个 BN 层, 每层 ~2048 维): 
# 每次 fwd+bwd 额外通信 ~50 × 2 × 2048 × 4 bytes = 800 KB — 可忽略
```

### 1.6 学习率如何随 GPU 数调整

| 缩放策略 | 公式 | 适用场景 |
|----------|------|----------|
| **Linear Scaling** | lr = base_lr × k (k=GPU数) | 小规模 (k ≤ 64) |
| **Sqrt Scaling** | lr = base_lr × √k | 中规模 |
| **Warmup + Linear** | 先 warmup 再 linear | 大规模通用 |
| **LARS/LAMB** | 逐层自适应 | 超大 batch (>32K) |

**为什么需要线性缩放？** 直观理解：
```
单 GPU: batch_size=32, 1 step = 1 次梯度更新
8 GPU:   batch_size=256, 1 step 看到的样本量是 8 倍
        → 等价的梯度"信噪比"更高 → 可以用更大的步长
```

实际案例（GPT-3 论文）：
```
batch_size 从 32K → 3.2M tokens
学习率从 6e-5 warmup 到 6e-4（约 10x，但 batch 是 100x）
→ 不是严格的 Linear Scaling，有单独的调参
```

### 1.7 Data Parallel 通信量精算

假设模型参数量 P，每个参数 FP16 梯度 = 2 bytes：

| 方法 | 每步通信量/GPU | N=8 时 |
|------|---------------|--------|
| Parameter Server (PS) | 发送 1×P×2 + 接收 1×P×2 = 4P bytes | 4P |
| Ring AllReduce | 2(N-1)/N × 2P ≈ 4P bytes | 3.5P |
| Tree AllReduce | 2×log₂N × 2P bytes | 12P |

**对 LLaMA-7B (P=7B)**：
```
Ring AllReduce 每次通信 = 2 × 7/8 × 7B × 2 bytes ≈ 24.5 GB / GPU
A100 NVLink 300 GB/s → 理论 0.08s（实际约 0.1-0.2s）
如果 step 耗时 2s → 通信占 ~10%
```

---

## 2. Pipeline Parallel（流水线并行）

### 2.1 问题场景

LLaMA-70B 有 80 层 Transformer，单卡装不下 → 把 80 层切成 4 段，每段 20 层放到不同 GPU。

```
GPU 0: Layer 0-19   (Embedding + 前20层)
GPU 1: Layer 20-39
GPU 2: Layer 40-59
GPU 3: Layer 60-79  (最后20层 + LM Head)
```

### 2.2 朴素流水线 —— 巨大的气泡

```
Time →

GPU 0: [F0][F0][F0][F0][B0][B0][B0][B0]                    ← 忙
GPU 1:    [F1][F1][F1][F1][B1][B1][B1][B1]                 ← 忙
GPU 2:       [F2][F2][F2][F2][B2][B2][B2][B2]              ← 忙
GPU 3:          [F3][F3][F3][F3][B3][B3][B3][B3]           ← 忙

F = Forward, B = Backward, 空白 = 空闲

空闲时间占 (N-1)/N = 3/4 = 75%  ← 只有 25% 利用率！
```

### 2.3 GPipe —— 用微批次填充气泡

把一个 mini-batch 切成 K 个 micro-batch：

```
K=4, N=4 (每个 GPU 负责 4 层)：

GPU 0: [F0][F1][F2][F3]              [B0][B1][B2][B3]
GPU 1:    [F0][F1][F2][F3]         [B0][B1][B2][B3]
GPU 2:       [F0][F1][F2][F3]    [B0][B1][B2][B3]
GPU 3:          [F0][F1][F2][F3][B0][B1][B2][B3]

气泡率 = (N-1)/(N+K-1) = 3/(3+4) = 3/7 ≈ 43%
K 越大，气泡越小
```

**但是**：K 越大 → 每个 micro-batch 越小 → GPU 利用率降低（SM 不饱和）

### 2.4 1F1B（One-Forward-One-Backward / PipeDream 风格）—— 进一步减少气泡

GPipe 要等所有 Forward 完成才开始 Backward → **激活需要全部缓存**。

1F1B：Forward 做几个 micro-batch 后，开始交替 Forward 和 Backward：

```
Time →

GPU 0: [F0][F1][F2][F3][B0]   [B1]   [B2]   [B3]   ← warmup 4, 然后 1F1B
GPU 1:    [F0][F1][F2][F3][B0]   [B1]   [B2]   [B3]
GPU 2:       [F0][F1][F2][F3][B0]   [B1]   [B2]   [B3]
GPU 3:          [F0][F1][F2][F3][B0]   [B1]   [B2]   [B3]

Warmup = N-1 个 micro-batch 的 Forward
然后稳定态：1 个 F + 1 个 B 交替
```

**好处**：
- 激活存储只需 (N-1) 个 micro-batch 而非 K 个 → 显存大幅节省
- 气泡仍存在，但比 GPipe 略高一点点

**代价**：
- 每个 micro-batch 的权重版本不一致（warmup 阶段和稳态阶段用的权重不同）
- 需要存多个权重版本（weight stashing），增加显存

### 2.5 气泡率推导（考试重点）

**GPipe**：
```
总时间槽数 = N + K - 1（FWD）+ N + K - 1（BWD）= 2(N+K-1)
其中空闲槽 = 2(N-1)
气泡率 = (N-1)/(N+K-1)
```

**1F1B（PipeDream-Flush）**：
```
Warmup: N-1 个 FWD
稳态: K-(N-1) 个 1F1B 对
Flush: N-1 个 BWD

总气泡 ≈ 2(N-1)，和 GPipe 差不多
但显存节省（只存 N-1 个 micro-batch 的激活 vs K 个）
```

### 2.6 负载均衡问题

不是所有层计算量相同：

```python
# 典型的 Transformer Block 计算量分布
Attention 部分: 4 × hidden² 次乘法
MLP 部分:      8 × hidden² 次乘法  ← 2倍于 Attention
总每层:       12 × hidden² 次乘法

如果切得不均匀（比如某 GPU 多分了一层），就会成为瓶颈
→ 所有 GPU 等最慢的那个
```

**解法**：按 **FLOPs** 而非层数来切分。可以用 profiling 工具（如 PyTorch Profiler）实测每层耗时，然后做负载均衡分配。

---

## 3. Tensor Parallel（张量并行 / Intra-layer）🔥 考试重点

### 3.1 Megatron-LM 风格的切分

以 Transformer 的 Self-Attention 为例。原始计算：

```
X: [b, s, h]  (batch=1, seq_len=s, hidden_dim=h)

Q = X × W_Q   W_Q: [h, h]
K = X × W_K   W_K: [h, h]
V = X × W_V   W_V: [h, h]

Attention(Q, K, V) = softmax(QK^T/√d) × V
Output = Attn × W_O   W_O: [h, h]
```

**切分方案（2 GPU, TP=2）**：

```
┌──────────────────────────────────────────────────────────────┐
│                        Self-Attention                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  X [b,s,h]                                                   │
│      │                                                       │
│      ├──────────────────┬──────────────────                  │
│      │   f (identity)   │   f (identity)                     │
│      ▼                  ▼                                    │
│  ┌─────────┐      ┌─────────┐                                │
│  │ W_Q1    │      │ W_Q2    │  ← Column-wise: W_Q = [W_Q1│W_Q2]
│  │ W_K1    │      │ W_K2    │    每列 h/2 维度                │
│  │ W_V1    │      │ W_V2    │                                │
│  └────┬────┘      └────┬────┘                                │
│       │                │                                      │
│  QKV1 [b,s,3h/2] QKV2 [b,s,3h/2]                             │
│       │                │                                      │
│   Self-Attn 1     Self-Attn 2    ← 各自独立算注意力            │
│       │                │                                      │
│  O1 [b,s,h/2]     O2 [b,s,h/2]                                │
│       │                │                                      │
│       │    AllReduce   │         ← 唯一一次通信！              │
│       └───────┬────────┘                                      │
│               ▼                                               │
│          O [b,s,h]                                            │
│               │                                               │
│          W_O [h,h]    ← Row-wise: 继续切                      │
│               │                                               │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 通信量精算 —— 用矩阵维度说话

设 hidden_dim = h, 分给 N 个 GPU：

**Column-wise（切 W 的列）**：

```
W: [h, h/N]
X: [b, s, h]  ← 不切，每个 GPU 都有完整输入

每个 GPU 算：Y_i = X × W_i  →  [b, s, h/N]
需要把 N 个 [b, s, h/N] 拼成 [b, s, h] → AllReduce

Forward 通信量: 2(N-1)/N × b×s×h × 2 bytes  (Ring AllReduce)
               ≈ 4 × b×s×h bytes  (N 较大时)
Backward 通信量: 同理 ≈ 4 × b×s×h bytes
```

**Row-wise（切 W 的行）**：

```
W: [h/N, h]
X: [b, s, h/N]  ← 输入也需要切！

每个 GPU 算：Y_i = X_i × W_i  →  [b, s, h]  ← 注意输出是完整的 h！

Forward: 需要 AllReduce 求和 N 个 [b, s, h]
         通信量 ≈ 4 × b×s×h bytes
Backward: 同理
```

**关键洞察**：无论 Row 还是 Column，每层的**通信量都正比于 b×s×h × 4 bytes**。

### 3.3 交替切分（Alternative Partitioning）—— 为什么能省一半

```
Layer K (Column-wise):
  Forward:  AllReduce ← 输入不切，输出要加
  结果: [b, s, h]（完整）

Layer K+1 (Row-wise):
  Forward:  不需要通信！← 输入 [b, s, h] 完整，切就行
  结果: [b, s, h]（输出需要加回来）
  → 但这里输出又需要 AllReduce

两层合起来：每个 Forward 需要 1 次 AllReduce
如果不交替（两层都 Column-wise）: 每个 Forward 需要 2 次 AllReduce
→ 节省 50% 通信
```

**Megatron-LM Transformer Block 的交替设计**：

```python
# Megatron-LM 的一个 Transformer Block 内通信
# TP=2，隐藏层维度 h=4096

Layer 1: QKV Projection  → Column-wise → fwd AllReduce (1)
Layer 2: Self-Attention   → 无参数，纯计算
Layer 3: Output Proj      → Row-wise    → fwd AllReduce (2)

Layer 4: MLP FC1 (h→4h)  → Column-wise → fwd AllReduce (3)
Layer 5: GELU            → 无参数
Layer 6: MLP FC2 (4h→h)  → Row-wise    → fwd AllReduce (4)

每个 Block: 4 次 AllReduce  → 48 层: 192 次 AllReduce
每次 AllReduce: 2(N-1)/N × b×s×h × 2 bytes
```

### 3.4 MLP 切分详解

MLP: `X → FC1 → GELU → FC2 → Y`

```
尺寸: X [b,s,h], W1 [h, 4h], W2 [4h, h]

Column-wise 切 W1:
  GPU 0: W1_0 [h, 2h], GPU 1: W1_1 [h, 2h]
  输出: [b,s,2h] 每个 GPU
  需要做 GELU（每个 GPU 独立）
  GELU 输出: [b,s,2h]

Row-wise 切 W2:
  GPU 0: W2_0 [2h, h], GPU 1: W2_1 [2h, h]
  GPU 0 算: [b,s,2h] × [2h,h] = [b,s,h]
  GPU 1 算: [b,s,2h] × [2h,h] = [b,s,h]
  
  Forward: AllReduce 两个 [b,s,h] 求和
```

**GELU 不需要通信**：因为没有可学习参数，且操作是逐元素的。

### 3.5 TP 通信代价的实战评估

LLaMA-70B, h=8192, bsz=8, seq_len=4096, TP=4:

```
单层 Attention 通信量:
  QKV column → AllReduce: 2(N-1)/N × 8×4096×8192 × 2 ≈ 1.5 GB
  Output row  → AllReduce: 同上 ≈ 1.5 GB
  FC1 column  → AllReduce: 2(N-1)/N × 8×4096×32768 × 2 ≈ 6 GB
  FC2 row     → AllReduce: 同上 ≈ 6 GB

单层通信量 ≈ 15 GB
80 层, 带宽 300 GB/s (NVLink) → 15×80/300 ≈ 4s 纯通信时间
当单层计算 ≈ 0.05s → 80 层 ≈ 4s → 通信占 50%！

结论：TP 适合在高速互联的同一节点内使用（NVLink/NVSwitch）
     跨节点 TP 性能很差（受限于网络带宽）
```

---

## 4. Sequence Parallelism（序列并行）

### 4.1 为什么还需要 Sequence Parallel？

Tensor Parallel 中，LayerNorm 和 Dropout 的输入输出**不做切分**，每个 GPU 存完整副本：

```
LayerNorm 的输入: [b, s, h]  ← 每个 GPU 都存一份！
Dropout 的输入:   [b, s, h]  ← 又存一份！

长序列场景 (s=8192): 
  单层 LayerNorm + Dropout 激活 = 2 × 8 × 8192 × 4096 × 2 = 1 GB
  80 层 → 80 GB  → 浪费！
```

### 4.2 切分方案

TP 区域之外，把序列维度也切掉：

```
Without SP:
GPU 0: [b, s, h] → LayerNorm → [b, s, h]  (完整副本)
GPU 1: [b, s, h] → LayerNorm → [b, s, h]  (完整副本)

With SP (TP=2, SP=2):
GPU 0: [b, s/2, h] → LayerNorm → [b, s/2, h]
GPU 1: [b, s/2, h] → LayerNorm → [b, s/2, h]
         ↑ 激活减半！                          ↑

进入 Attention 区域前: AllGather → 恢复 [b, s, h] (通过 TP 通信顺带完成)
离开 Attention 区域后: Reduce-Scatter → 切回 [b, s/2, h]
```

**和 TP 通信融合**：Sequence Parallel 的 AllGather/Reduce-Scatter 可以和 TP 的通信**复用同一通信操作**，不增加额外开销。

---

## 5. ZeRO（Zero Redundancy Optimizer）

### 5.1 显存四部分

```
训练状态下单 GPU 显存分布：
┌─────────────────────────────────────────────┐
│ Model States (模型状态)                       │
│ ┌─────────┬─────────┬─────────────────────┐ │
│ │ Params  │ Grads   │ Optimizer States    │ │
│ │ (参数)   │ (梯度)   │ (Adam: m, v, fp32)  │ │
│ └─────────┴─────────┴─────────────────────┘ │
│                                             │
│ Residual States (残留状态)                    │
│ ┌──────────────────┬──────────────────────┐ │
│ │ Activations      │ Temp Buffers         │ │
│ │ (激活，可重计算)   │ (临时缓冲区)           │ │
│ └──────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────┘
```

以 LLaMA-7B, FP16 训练为例：

| 组件 | 每 GPU (DP=8) | 说明 |
|------|---------------|------|
| Params (fp16) | 14 GB | 7B × 2 bytes |
| Grads (fp16) | 14 GB | 同上 |
| Optimizer States (fp32) | 84 GB | 7B × 3 × 4 bytes |
| Activations | ~20 GB | seq 2048, bsz 8 |
| **总计** | **~132 GB** | **远超单卡 80GB！** |

### 5.2 ZeRO-1：只切 Optimizer States

```
DP=4 时：
GPU 0: 完整 Params + 完整 Grads + Optimizer States[0/4]
GPU 1: 完整 Params + 完整 Grads + Optimizer States[1/4]
...

Step 流程:
1. FWD + BWD (和 DP 一样)
2. AllReduce Gradients (和 DP 一样)
3. Reduce-Scatter → 每个 GPU 只拿自己那 1/4 optimizer states
4. 各自 update 自己的 1/4 参数
5. AllGather → 拼回完整参数

显存节省: Optimizer States 从 84GB → 84/N GB
通信增加: 多了一次 Reduce-Scatter + AllGather (但 optimizer update 通信量很小)
```

### 5.3 ZeRO-2：再切 Gradients

```
在 ZeRO-1 基础上:
- Gradients 也切分存储 → 每个 GPU 只存 1/N 的梯度
- AllReduce 变为 Reduce-Scatter (梯度不需要在每个 GPU 上完整)

显存节省: Grads 从 14GB → 14/N GB
```

### 5.4 ZeRO-3：连 Parameters 也切

```
终极方案: 连模型参数都分片

每个 GPU 在任何时刻只持有 1/N 的参数
需要用到某个参数时 (FWD/BWD)，通过 AllGather 临时收集

FWD 流程:
  Layer 1 参数在 GPU 0:
    GPU 0: Broadcast → 其他 GPU 拿到 Layer 1 参数 → 全体做 FWD
  Layer 2 参数在 GPU 1:
    GPU 1: Broadcast → 其他 GPU 拿到 Layer 2 参数 → 全体做 FWD
  ...

显存节省: Params 从 14GB → 14/N GB
通信代价: 每个 layer 的 AllGather/Broadcast（和模型结构耦合）
```

### 5.5 三种 ZeRO 汇总

| | Standard DP | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| Params | N副本 | N副本 | N副本 | **分片** |
| Grads | N副本 | N副本 | **分片** | **分片** |
| Optim States | N副本 | **分片** | **分片** | **分片** |
| 显存 (7B, N=8) | 132 GB | 121 GB | 119 GB | **17 GB** |
| 通信量 vs DP | 1× | ~1.5× | ~1.5× | ~2× |
| 实现难度 | 简单 | 中等 | 中等 | 复杂 |

> ZeRO-3 让 7B 模型在 8×A100 上从 132GB 降到 17GB → 单卡 80GB 绰绰有余

### 5.6 FSDP（PyTorch 的 ZeRO-3 实现）

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = FSDP(
    model,
    sharding_strategy="FULL_SHARD",  # ZeRO-3
    # sharding_strategy="SHARD_GRAD_OP",  # ZeRO-2
    # sharding_strategy="NO_SHARD",       # Standard DP
)

# 和 DDP 一样的 API
for data, label in dataloader:
    loss = model(data)
    loss.backward()
    optimizer.step()
```

---

## 6. Gradient Checkpointing（激活重计算 / 梯度检查点）

### 6.1 思想

```
不做 Checkpointing:
FWD: 存所有激活 → BWD: 直接用 → 显存爆

做 Checkpointing:
FWD: 只存"检查点"的激活 → 其他激活扔掉
BWD: 需要某段激活时 → 从最近的检查点重新 FWD 算出来

时间换空间：多跑一次 FWD，省掉大部分激活存储
```

### 6.2 PyTorch 一行代码

```python
from torch.utils.checkpoint import checkpoint

# 标记哪些层做重计算
def custom_block(x):
    x = self.attention(x)
    x = self.mlp(x)
    return x

x = checkpoint(custom_block, x)  # 该 block 不存激活
```

**对 Transformer 标准做法**：每个 Transformer Block 做一个 checkpoint。

显存从 O(n_layers) → O(sqrt(n_layers)) 或 O(1)（取决于检查点放置策略）。

### 6.3 代价分析

- 多一次完整的 Forward 计算 → step 时间增加约 **33%**
- 显存节省 50-80%（取决于模型结构）
- 对大模型来说，**用时间换显存非常划算**——省下的显存可以增大 batch size，反而提升吞吐

---

## 7. 混合精度训练（Mixed Precision Training）

### 7.1 为什么需要混合精度

```
FP32: 4 bytes/参数 → 精度高，但贵
FP16: 2 bytes/参数 → 省一半显存和带宽，但有精度问题
BF16: 2 bytes/参数 → 和 FP32 一样的指数范围，适合训练
```

### 7.2 FP16 训练的三大问题与解法

| 问题 | 原因 | 解法 |
|------|------|------|
| 梯度下溢 | 梯度太小 (< 2⁻²⁴) → FP16 直接变 0 | **Loss Scaling** |
| 权重更新被吞 | weight − lr×grad 差值太小 | FP32 Master Weights |
| 精度不够 | FP16 尾数只有 10 bits | 关键操作留 FP32 |

### 7.3 标准混合精度流程

```python
# 标准 FP16 混合精度训练的一步

# 1. 维护 FP32 的主权重
master_weights = fp32_copy(model_params)

# 2. Forward: FP16
fp16_weights = master_weights.to(fp16)
output = model.forward_fp16(input, fp16_weights)
loss = criterion(output)

# 3. Loss Scaling: 把 loss × scale_factor（如 1024）
scaled_loss = loss * loss_scale

# 4. Backward: FP16 梯度
scaled_loss.backward()  # grad = scaled_grad

# 5. Unscale: 梯度 ÷ scale_factor
grad = scaled_grad / loss_scale  # → FP32

# 6. 更新 FP32 主权重
master_weights = optimizer.step(master_weights, grad)
```

### 7.4 BF16 的优势

```
FP32: [S(1) | Exp(8) | Mantissa(23)]
FP16: [S(1) | Exp(5) | Mantissa(10)]  ← 最大 65504，梯度容易溢出
BF16: [S(1) | Exp(8) | Mantissa(7)]   ← 和 FP32 一样的范围！

BF16 不需要 Loss Scaling，直接训！
但精度低 → 部分算子仍需 FP32（如 softmax, layernorm）
```

**实际选择**：
- NVIDIA A100/H100 → BF16 首选（有硬件支持）
- NVIDIA V100 → FP16 + Loss Scaling
- Google TPU → BF16（原生支持）

---

## 8. 通信原语详解

### 8.1 四大集体通信原语

| 原语 | 输入 | 输出 | 操作 |
|------|------|------|------|
| **AllReduce** | 每人有一个 tensor | 每人有 sum(tensors) | 全局归约 |
| **AllGather** | 每人有一个 tensor | 每人有 concat(tensors) | 全局收集 |
| **Reduce-Scatter** | 每人有一个 tensor | 每人有 sum 后的 1/N | 归约+分发 |
| **Broadcast** | 1 人有 tensor | 每人有该 tensor | 一对多广播 |

### 8.2 AllReduce = Reduce-Scatter + AllGather

这是 Ring AllReduce 的核心洞察：

```
AllReduce([a,b,c,d] 在 4 GPU):

Phase 1 — Reduce-Scatter (3 步):
  GPU0: a → +b₃ → +c₂ → 最终持有 Σc (chunk 3 的累加)
  GPU1: b → +c₀ → +d₃ → 最终持有 Σd (chunk 0 的累加)
  每个 GPU 最终持有 1/4 的归约结果

Phase 2 — AllGather (3 步):
  把各自的 1/4 归约结果广播给所有人
  → 每人都有 [Σa, Σb, Σc, Σd]
```

### 8.3 通信量公式

设数据总量为 D，N 个 Worker：

| 操作 | 理想通信量/Worker | 延迟步数 |
|------|-------------------|----------|
| AllReduce (Ring) | 2(N-1)/N × D | 2(N-1) |
| AllReduce (Tree) | 2 × D | 2×log₂N |
| AllGather (Ring) | (N-1)/N × D | N-1 |
| Reduce-Scatter (Ring) | (N-1)/N × D | N-1 |
| Broadcast (Tree) | D | log₂N |

**为什么 Ring 在 N 大时接近最优？**

```
Ring AllReduce: 通信量 → 2D (当 N→∞)
Tree AllReduce: 通信量 → 2D (但每步要多发)
Parameter Server: 通信量 → 2D (但 PS 是瓶颈)

Ring 的好处：每个 Worker 只连两个邻居 → 链路利用率 100%
```

---

## 9. 3D 并行 —— 真实大模型训练

### 9.1 三维并行网格

```
           DP (数据并行)
           │
     ┌─────┼─────┐
    GPU0  GPU1  GPU2  ← DP group 0
    GPU3  GPU4  GPU5  ← DP group 1
           │
           └── TP (张量并行) = 2
     ┌─────┼─────┐
    GPU0  GPU3  GPU6  ← PP stage 0
    GPU1  GPU4  GPU7  ← PP stage 1
    GPU2  GPU5  GPU8  ← PP stage 2
           │
           └── PP (流水线并行) = 3

总共需要: PP=3, TP=2, DP=? 
  每节点内 NVLink → 适合 TP
  节点间 InfiniBand → 适合 DP/PP
```

### 9.2 GPT-3 的真实配置

| 模型 | 参数 | GPU | PP | TP | DP | Batch Size |
|------|------|-----|----|----|-----|-----------|
| GPT-3 Small | 125M | — | 1 | 1 | — | 0.5M tokens |
| GPT-3 Medium | 350M | — | 1 | 1 | — | 0.5M tokens |
| GPT-3 Large | 760M | — | 1 | 1 | — | 0.5M tokens |
| GPT-3 XL | 1.3B | — | 1 | 1 | — | 1M tokens |
| GPT-3 2.7B | 2.7B | — | 1 | 1 | — | 1M tokens |
| GPT-3 6.7B | 6.7B | — | 1 | 1 | — | 2M tokens |
| GPT-3 13B | 13B | — | 1 | 1 | — | 2M tokens |
| **GPT-3 175B** | **175B** | **10K V100** | **8** | **8** | **64** | **3.2M tokens** |

```
总 GPU 数 = PP × TP × DP = 8 × 8 × 64 = 4096（训练用了约 10K V100，含冗余）
```

### 9.3 LLaMA 2 70B 的训练配置

```
LLaMA 2 70B on 2000 A100-80GB:

PP=1 (模型大小 70B，80GB × TP=8 能装下 8 层)
TP=8 (节点内 NVSwitch 900 GB/s)
DP=250 (250 × 8 = 2000 GPU)

每 GPU 的 micro_batch_size = 1
gradient_accumulation_steps = 16
→ global_batch_size = 2000 × 1 × 8 / 8 × 16 = 4M tokens  (≈ DP×micro_bsz×accum)
```

### 9.4 并行策略的选择决策树

```
模型能装进单卡吗？
├── YES → Data Parallel
└── NO
    ├── 跨节点带宽 < 20 GB/s (如 100Gbps Ethernet)?
    │   └── Pipeline Parallel（通信少，一次只传激活）
    │
    ├── 同一节点内 NVLink > 300 GB/s?
    │   └── Tensor Parallel（通信多但带宽够）
    │
    ├── 隐藏层 > 8192 且序列长度 > 4096?
    │   └── 加 Sequence Parallel（省激活）
    │
    └── 优化器状态是瓶颈？
        └── ZeRO-1/2/3 或 FSDP
```

---

## 10. 真实故障与调试

### 10.1 常见问题

| 症状 | 原因 | 排查方法 |
|------|------|----------|
| loss 不下降 | LR 太小，或 bs 太大没调 LR | 先看单卡能收敛吗 |
| OOM | 激活太大 | 加 checkpointing；或检查是否有 tensor 泄漏 |
| 训练速度不随 GPU 线性增长 | 通信瓶颈 | `torch.profiler` 看通信占比 |
| 不同 GPU loss 不同 | 随机种子没同步 | `torch.manual_seed(seed)` 所有 rank 都要设 |
| NCCL 超时 | 某 GPU 卡住或带宽不足 | `NCCL_DEBUG=INFO` 查看日志 |
| 某些 step 突然很慢 | GC 触发，或 checkpoint 保存 | 查看 jitter 模式 |

### 10.2 通信瓶颈的快速诊断

```python
# PyTorch Profiler 看每层的通信/计算时间
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    for _ in range(10):
        loss = model(data)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

---

## 11. 核心公式速查

| 公式 | 说明 | 例子 |
|------|------|------|
| 气泡率 = (N-1)/(N+K-1) | GPipe 空闲率 | N=4, K=4 → 43% |
| Ring AR 通信量 = 2(N-1)/N × D | 每 Worker 通信量 | N=8 → 1.75×D |
| 梯度累积等价 batch = per_gpu_bsz × accum_steps × dp_size | 全局 batch size | 4 × 8 × 64 = 2048 |
| ZeRO-3 参数显存 = P×2/N | 参数分片 (FP16) | 7B, N=8 → 1.75GB |
| TP 每层通信 = 4×bsz×seq×hidden×2 bytes | 每层每步通信 | 参考 3.5 节 |
| 激活显存 ≈ bsz×seq×hidden×(34+5×heads×seq/hidden)×2 | Transformer 激活 | 参考开篇例子 |

---

!!! danger "考试重点"
    1. Tensor Parallel：Row-wise vs Column-wise 各自的 Forward/Backward 通信原语
    2. Alternative Partitioning：为什么能减少同步（每两层一次 AllReduce）
    3. Ring AllReduce：步数 2(N-1)，每步通信量 1/N
    4. GPipe vs 1F1B：气泡率推导，为什么 1F1B 省显存
    5. ZeRO 三阶段的显存节省和通信代价
    6. 四种并行的通信模式对比（能不能 overlap，适合什么拓扑）
