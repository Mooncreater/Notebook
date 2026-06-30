---
comments: true
---

# 并行训练（Parallel Training）

!!! tip "核心要点"
    大模型训练的四种并行策略：Data Parallel（数据并行）→ Model Parallel（模型并行）→ Pipeline Parallel（流水线并行）→ Tensor Parallel（张量并行）。考试重点是 Tensor Parallel 的切分方式与通信模式。

## 为什么需要分布式训练？

- 模型越来越大（GPT-3: 175B 参数），单 GPU 装不下
- 训练数据越来越大，单 GPU 算不完
- GPT-3 训练显存需求：Optimizer States ~3259 GB，远超单卡容量

**显存占用估算（Transformer 模型）**：

| 组件 | 大小 |
|------|------|
| 参数 (param) | `12 × hid²` （每层） |
| 激活 (activation) | `20 × bsz × seq × hid + n_h × bsz × seq × seq` |
| Embedding param | `voc × hid` |
| 混合精度训练 | FP32: param + grad + optimizer (momentum+variance) = 4×4 bytes |
| | FP16: param + grad = 2×2 bytes |
| | Activation: 全 FP16 |

---

## 并行策略分类

```
Parallel Training
├── Data Parallel（数据并行）
└── Model Parallel（模型并行）
    ├── Pipeline Parallel（层间并行 / Inter-layer）
    └── Tensor Parallel（层内并行 / Intra-layer）
```

---

## 1. Data Parallel（数据并行）

**思路**：每个 Worker 持有完整模型副本，分配不同 mini-batch 数据。

**流程**：
- Forward: 每个 Worker 独立计算
- Backward: 每个 Worker 独立计算梯度
- **AllReduce 梯度**：所有 Worker 同步梯度后取平均
- 更新权重（所有 Worker 得到相同的权重）

**通信**：AllReduce of gradients
- 可与计算重叠（overlap）

**挑战**：
- Strong scaling（增加 Worker，固定 batch size）：BN 等层要求最小 batch size ≥ 16
- Weak scaling（增加 batch size）：需调整超参（学习率等），达到相同精度需要更多 epochs

---

## 2. Pipeline Parallel（流水线并行 / Inter-layer）

**思路**：按层切分模型，Worker 0 负责 Layer 1-2，Worker 1 负责 Layer 3-4...

```
Worker 0: Layer 1 → Layer 2
    ↓ (activation)
Worker 1: Layer 3 → Layer 4
    ↓ (activation)
Worker 2: Layer 5
```

**问题：Idle Bubbles（空闲气泡）**
- 无优化时：N 个 Worker，空闲率 = (N-1)/N（如 3 Worker = 67% 空闲）
- **GPipe 优化**：用 Sub-minibatches 填充气泡
  - K 个 sub-minibatch，空闲率 = (N-1)/(N+K-1)
  - K=N → 50%；K=4N → 20%

**通信**：Point-to-Point 通信 activation（forward）和 activation gradient（backward）
- 难以与计算重叠
- 负载均衡困难（不同层计算量不同）

---

## 3. Tensor Parallel（张量并行 / Intra-layer）🔥 考试重点

**思路**：在**同一层内部**切分权重矩阵，多个 Worker 协同完成一层计算。

### 3.1 按行切分（Row-wise Partitioning）

将权重矩阵 W **按行切分**给各 Worker：

```
W = [W₀; W₁; W₂]    ← 每个 Worker 持有一部分行

Forward:  Yᵢ = Wᵢ × X    （各 Worker 独立计算部分输出）
          Y = [Y₀; Y₁; Y₂]  ← 需要 AllGather 拼起来
```

- **Forward 通信**：**AllGather**（收集各 Worker 的部分输出，拼成完整 Y）
- **Backward 通信**：**Reduce-Scatter**（各 Worker 有完整 ∂L/∂Y，各自取对应行的梯度求和）

### 3.2 按列切分（Column-wise Partitioning）

将权重矩阵 W **按列切分**给各 Worker：

```
W = [W₀ | W₁ | W₂]    ← 每个 Worker 持有一部分列
X = [X₀; X₁; X₂]      ← 输入也要对应切分

Forward:  Y = W₀×X₀ + W₁×X₁ + W₂×X₂  ← 需要 AllReduce
```

- **Forward 通信**：**AllReduce**（每个 Worker 的部分积求和得到完整 Y）
- 等价于：Reduce-Scatter（或不通信直接发部分积然后 AllReduce）

实际上：
- **Forward 通信**：**Reduce-Scatter**（各 Worker 的部分积分发并求和）
- **Backward 通信**：**AllGather**（各 Worker 需要完整 ∂L/∂X）

### 3.3 通信模式总结

| 切法 | Forward 通信 | Backward 通信 |
|------|-------------|--------------|
| **Row-wise** | **AllGather**（拼输出） | **Reduce-Scatter**（散梯度） |
| **Column-wise** | **Reduce-Scatter**（求和输出） | **AllGather**（拼梯度） |

!!! tip "记忆技巧"
    Row-wise Forward 需要 AllGather（因为是"分着算，拼起来"）；Column-wise Forward 需要 Reduce-Scatter（因为是"分着算，加起来"）。Backward 正好反过来。

### 3.4 Alternative Partitioning（交替切分）🔥 重要！

**关键想法**：相邻层交替使用 Row-wise 和 Column-wise，**减少同步次数**！

```
Layer K:   Row-wise partitioning（每 Worker 独立输出 = 下一层输入）
Layer K+1: Column-wise partitioning（每 Worker 输入已经是分好的！）
  → 不需要通信！
  
Layer K+1: Column-wise partitioning
Layer K+2: Row-wise partitioning
  → 需要 AllReduce（因为 Column-wise 输出是完整的）
```

**效果**：
- 不交替：每层都需要通信（fwd 和 bwd 各一次）
- 交替：**每两层通信一次**（AllReduce），同步减半

**Row↔Col 交替规则**：
- Row-wise 在 Forward 是 AllGather → 在 Backward 是 Reduce-Scatter
- Col-wise 在 Forward 是 Reduce-Scatter → 在 Backward 是 AllGather
- 交替后：每两层做一次 AllReduce（Forward + Backward 各一次）

### 3.5 Transformer 中的 Tensor Parallel

Transformer Block 的典型切分：
```
Attention:
  QKV Projection: Column-wise（切 Q/K/V 的列）
  Output Projection: Row-wise（切输出的行）

MLP:
  FC1 (h→4h): Column-wise
  FC2 (4h→h): Row-wise
```

---

## 4. AllReduce 通信详解 🔥 考试重点

AllReduce 的目标：所有 Worker 对各自持有的数据求和（或求平均），**每个 Worker 都得到完整结果**。

### 4.1 Ring AllReduce（百度）

**拓扑**：Worker 组成环形，每个只与两个邻居通信。

**两阶段**：

**Phase 1: Reduce-Scatter（N-1 步）**
- 将数据分成 N 块
- 第 k 步：每个 Worker 发送一块给下游邻居，接收上游一块并累加
- 经过 N-1 步后，每个 Worker 持有一块**完整归约结果**

**Phase 2: AllGather（N-1 步）**
- 第 k 步：每个 Worker 发送自己的归约块给下游邻居
- 经过 N-1 步后，每个 Worker 都有全部 N 块的完整结果

**关键参数**：

| 参数 | 数值 |
|------|------|
| 总步数 | **2(N-1)** |
| 每步通信量 | **1/N 的数据量** |
| 总通信量（每 Worker） | **2(N-1)/N × Data** ≈ 2×Data |
| 同步次数 | **2(N-1)** 次 |

**示例（N=4，数据 = [a,b,c,d]）**：

```
Reduce-Scatter:
  Step 1: GPU0 发 d→GPU1, GPU1 发 a→GPU2, GPU2 发 b→GPU3, GPU3 发 c→GPU0
  Step 2: 继续传 + 累加
  Step 3: 完成，每个 GPU 持有一块完整归约

AllGather:
  Step 1-3: 每个 GPU 广播自己的归约块
  最终：所有 GPU 都有 [Σa, Σb, Σc, Σd]
```

### 4.2 In-Switch AllReduce

- 每个 Worker 与交换机通信
- **仅 1 步**，但每个 Worker 发送/接收 **全部 N 份数据**
- 所有 Worker lock-step 工作

### 4.3 两种 AllReduce 对比

| | Ring AllReduce | In-Switch AllReduce |
|--|---------------|-------------------|
| 步数 | 2(N-1) | 1 |
| 每步数据量 | 1/N | N倍 |
| 总通信量/Worker | ~2×Data | Data |
| 适合 | 任意拓扑有环即可 | 有交换机支持 |

---

## 5. 通信模式总对比

| 并行策略 | 通信模式 | 能否 overlap 计算 |
|----------|----------|------------------|
| Data Parallel | AllReduce (gradients) | ✅ 可以 |
| Pipeline Parallel | P2P (activations + grad) | ❌ 困难 |
| Tensor Parallel (Row) | Fwd: AllGather / Bwd: Reduce-Scatter | ❌ 困难 |
| Tensor Parallel (Col) | Fwd: Reduce-Scatter / Bwd: AllGather | ❌ 困难 |
| Tensor Parallel (交替) | **AllReduce 每两层** | ❌ 困难 |

---

## 6. ZeRO（Zero Redundancy Optimizer）

**核心思想**：每个 GPU 只存 Optimizer States 的一个子集，而非 Data Parallel 那样存完整副本。

- 显存节省：显著减少 optimizer states 冗余
- 代价：更多通信（更多 collectives）
- 效果：能训练更大的模型

---

## 7. 实际训练策略选择

| 场景 | 推荐策略 |
|------|----------|
| 模型能装进单卡 | Data Parallel |
| 模型装不进单卡 | Tensor Parallel (层内) + Pipeline Parallel (层间) |
| 超大模型 (GPT-3) | 三者结合：DP + TP + PP |
| 1K GPU/NPU, batch=1K | DP 为主，辅以 TP |
| 10K GPU/NPU, batch=1K | DP + TP + PP 混合 |

---

## 8. 并行训练公式记忆

- **Data Parallel 通信量**：每 Worker `2(N-1)/N × |Gradient|`（Ring AllReduce）
- **Pipeline idle 率**：`(N-1)/(N+K-1)`（N=Worker数，K=sub-minibatch数）
- **Tensor Parallel 通信**：不交替每层通信，交替每两层 AllReduce
- **AllReduce 轮数**：Ring = 2(N-1) 步，每步 1/N 数据

!!! danger "考试重点"
    1. Row-wise vs Column-wise 各自 Forward/Backward 的通信原语
    2. Alternative Partitioning 如何减少同步（每两层一次 AllReduce）
    3. Ring AllReduce 的轮数 2(N-1) 和每步通信量 1/N
    4. 四种并行的通信模式对比表
