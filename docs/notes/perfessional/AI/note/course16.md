---
comments: true
---

# 总复习（Overview）

!!! tip "考试须知"
    - 形式：**选择题 + 5 题简答题**
    - 全英文试卷，文字量大，**细心读题**
    - 可能有**作图题**，带铅笔和尺子
    - 复习脉络：以总复习 PPT 为纲，回溯各讲内容

---

## 考试范围与重点

PPT 1~14 + 16，**拓展性/前沿研究内容不考**。

!!! danger "划重点"
    1. Tomasulo：RAT 和 Reservation Station 的含义、更新时机和方式
    2. Parallel Training：Tensor Parallel 按行/列切参数、Alternative Partitioning、通信模式
    3. AllReduce：Ring AllReduce 的轮数 (N-1)$、每步通信量 /N$
    4. 四道例题：Roofline、Pipelined CPU、Performance Analysis、Cache

---

## 例题 1：Roofline Model

**题目要点**：给定计算平台的峰值算力和带宽，分析应用的 Arithmetic Intensity，判断瓶颈并计算可达到性能。

OI = \frac{FLOPs}{Bytes}

Attainable\ Performance = \min(Peak\_Compute,\ \ OI \times Peak\_Bandwidth)

- $ 高 → Compute Bound → 性能受限于算力 Peak
- $ 低 → Memory Bound → 性能受限于  \times Bandwidth$

**变体：Memory Roofline**（Cache 有不同的带宽上限）

| 存储层级 | 带宽特征 |
|----------|----------|
| DRAM | 有限带宽 |
| HBM | 中等带宽 |
| Cache | 大带宽 |

---

## 例题 2：Pipelined CPU (Tomasulo Performance Analysis)

**给定**：OoO 处理器，1 adder (2-cycle latency, pipelined) + 1 multiplier (4-cycle latency, pipelined)

**指令序列**：
`
I1  ADD  r3, r1, r2
I2  IMUL r4, r1, r3     ; RAW: 依赖 I1 的 r3
I3  IMUL r1, r3, r4     ; RAW: 依赖 I2 的 r4; WAR: 写 r1
I4  ADD  r4, r5, r3     ; WAW: 写 r4（与 I2 冲突）
I5  IMUL r6, r4, r5     ; RAW: 依赖 I4 的 r4
`

**解题要点**：

### (1) WAW Hazard 识别与解决

I2 写 r4，I4 也写 r4 → **WAW**（Write-After-Write 假依赖）

**解决方案**：寄存器重命名（Register Renaming）
- 用 ROB Entry ID 或 RAT Tag 替代架构寄存器名
- I2: r4 → ROB100, I4: r4 → ROB101 → 两者不再冲突

### (2) Dataflow Graph（作图题！）

`
r1 ──┬── I1(ADD) ── r3 ──┬── I2(IMUL) ── r4 ──┬── I3(IMUL) ── r1
r2 ──┘                    │                      │
r1 ───────────────────────┘                      │
r5 ──┬── I4(ADD) ── r4 ─────────────────────────┘
r3 ──┘                    │
r5 ───────────────────────┼── I5(IMUL) ── r6
r4(I4) ───────────────────┘
`

### (3) 流水线时序分析

**阶段**：F(etch) / D(ecode) / E(xecute) / R(Reorder) / W(riteback)

| Cycle | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------|---|---|---|---|---|---|---|---|---|----|----|----|
| I1 | F | D | E | E | R | W |   |   |   |    |    |    |
| I2 |   | F | D | - | - | - | E | E | E | E  | R  | W  |
| I3 |   |   | F | D | - | - | - | - | - | -  | E  | E  |
| I4 |   |   |   | F | D | E | E | R | W |    |    |    |
| I5 |   |   |   |   | F | D | - | - | - | -  | -  | E  |

**关键观察**：
- I4 不依赖 I2/I3（只需要 r3 和 r5），可**超越 I2/I3 提前执行**
- I2 在 cycle 3 D 阶段因等待 r3 而停顿，但 I3/I4/I5 **不阻塞发射**
- OoO 执行使总时间从 In-order 的 ~16 cycles 降至 ~12 cycles

---

## 例题 3：Performance Analysis（CPI + AMAT + CPU Time）

**给定**：
- 处理器 P，时钟周期 1ns
- Load=5, Store=3, Arithmetic=2, Branch=3 (cycles)
- 应用 A：20% load, 10% store, 50% arith, 20% branch

**(a) 理想 CPI**：

CPI_{ideal} = 0.2 \times 5 + 0.1 \times 3 + 0.5 \times 2 + 0.2 \times 3 = 1.0 + 0.3 + 1.0 + 0.6 = 2.9

**(b) 平均内存访问时间 AMAT**：

AMAT = HitTime + MissRate \times MissPenalty

直接映射 Cache, miss rate=1.4%, miss penalty=100ns, hit time=1 cycle=1ns：

AMAT = 1 + 0.014 \times 100 = 2.4ns

**(c) CPU Time（100 条指令，1.3 次访存/指令）**：

CPU\ Time = IC \times \left(CPI_{base} + \frac{MemAccesses}{IC} \times MissRate \times MissPenalty_{cycles}\right)

= 100 \times (2.9 + 1.3 \times 0.014 \times 100) = 100 \times (2.9 + 1.82) = 472\ cycles = 472ns

**(d) 比较 Direct-Mapped vs 2-way Set-Associative**：

| | Direct-Mapped | 2-way SA |
|--|--------------|----------|
| Miss Rate | 1.4% | 1.0% |
| Clock | 1ns | 1.05ns |

Time_{DM} = 100 \times (2.9 + 1.3 \times 0.014 \times 100) \times 1 = 472ns
Time_{2way} = 100 \times (2.9 + 1.3 \times 0.010 \times 100) \times 1.05 = 100 \times 4.2 \times 1.05 = 441ns

:arrow_right: **2-way set-associative 更快**（更低的 miss rate 的收益 > 时钟周期增加的代价）

---

## 例题 4：Cache 地址映射与性能

### 4.1 地址分解公式

| 参数 | 公式 |
|------|------|
| Block offset bits | $\log_2(block\_size)$ |
| Index bits | $\log_2(\frac{cache\_size}{block\_size \times associativity})$ |
| Tag bits | \_bits - index\_bits - offset\_bits$ |

**示例**：256-byte memory, 8-byte blocks, 64-byte direct-mapped cache (8 blocks)
- Offset = $\log_2 8 = 3$ bits
- Index = $\log_2 8 = 3$ bits  
- Tag =  - 3 - 3 = 2$ bits

### 4.2 Cache 三种组织方式对比

| 方式 | 优点 | 缺点 |
|------|------|------|
| **Direct-Mapped** | 实现简单，访问快 | Conflict miss 多（ping-pong: A,B,A,B.. → 0% hit） |
| **Fully-Associative** | 无 conflict miss | 比较器太多，不可扩展 |
| **Set-Associative** | **最优折中** | MUX 稍复杂，访问稍慢 |

### 4.3 Cache Miss 三种类型

| 类型 | 成因 | 解决方法 |
|------|------|----------|
| **Compulsory**（冷启动） | 首次访问 | Prefetching |
| **Capacity**（容量） | Cache 太小 | 更大 Cache / 软件管理 |
| **Conflict**（冲突） | 多块映射同一位置 | 提高关联度 / Victim Cache |

### 4.4 替换策略

1. 优先换 Invalid block
2. 否则按：**Random** / FIFO / **LRU**（Least Recently Used）
   - LRU 的问题：cyclic access 可能导致 0% hit（Set Thrashing）
   - 实际 Intel CPU 用 LRU+Random **混合策略**
   - 最优策略：Belady OPT（替换未来最远访问的块），**理论上无法实现**

### 4.5 写策略

| 策略 | 行为 | 特点 |
|------|------|------|
| **Write-Back**（默认） | 只写 Cache, evict 时写回 | 快，需 dirty bit |
| **Write-Through** | 同时写 Cache + 内存 | 简单但慢 |
| **Write-Allocate**（默认） | Miss 时先加载到 Cache | 配合 Write-Back |
| **Write-No-Allocate** | Miss 时直接写内存 | PCIe/IO 场景 |

### 4.6 指令 Cache vs 数据 Cache

- **分离**（L1 通常分离）：避免 I/D 相互 thrash，各阶段访问不同位置
- **统一**（L2/L3 通常统一）：动态共享空间，利用率更高

---

## Cache Coherence（缓存一致性）

### Coherence vs Consistency

| | Coherence | Consistency |
|--|-----------|-------------|
| 范围 | 同一地址 | 所有地址 |
| 关注 | 各核看到的同一地址**最后写入值一致** | 所有内存操作的**全局顺序** |

### MESI 协议

| 状态 | 含义 | 本地读/写 |
|------|------|----------|
| **M**(odified) | 仅本 Cache 有，已修改 | 直接读/写，无需总线操作 |
| **E**(xclusive) | 仅本 Cache 有，未修改 | 读无需总线；写→M，无需总线 |
| **S**(hared) | 多 Cache 共享 | 读无需总线；写→需 Invalidate 总线 |
| **I**(nvalid) | 不在 Cache | 读→Read Miss；写→Write Miss |

!!! tip "MESI vs MSI 关键区别"
    新增 **E 状态**：当 Cache 独占一个 clean block 时，写操作**无需总线广播**（MSI 需要），减少总线流量。

### 一致性实现方式

| 方式 | 机制 | 特点 |
|------|------|------|
| **Snoop（嗅探）** | 所有 Cache 监听总线 | 适合总线拓扑，小规模 |
| **Directory（目录）** | 目录记录每个 block 的共享者 | 适合大规模，每个 block 有 Home Node |

---

## 核心公式速查

| 公式 | 说明 |
|------|------|
|  = \frac{1}{(1-f)+f/n}$ | Amdahl 定律 |
| {max} = \frac{1}{1-f}$ | 极限加速比 |
|  = FLOPs / Bytes$ | 运算强度 |
|  = \min(Peak,\ OI \times BW)$ | Roofline 性能 |
|  = \lambda \times W$ | Little 定律 |
|  = HitTime + MissRate \times MissPenalty$ | 平均访存时间 |
|  = \sum(ratio_i \times cycle_i)$ | 平均 CPI |
| \ Time = IC \times CPI \times ClockCycle$ | CPU 时间 |
| AllReduce steps = (N-1)$ | Ring AllReduce 步数 |
| Pipeline idle = /(N+K-1)$ | GPipe 空闲率 |

---

## 考试答题策略

1. **选择题**：数好题数，别漏做
2. **简答题**：先读题再答题，英文题干可能很长但关键信息会交代清楚
3. **作图题**：用铅笔+尺子，可能涉及 Dataflow Graph、流水线时序图、Cache 地址分解
4. **计算题**：写清楚步骤，带单位
5. **心态**：阅读量大时保持耐心，所有需要的数据题目都会给
