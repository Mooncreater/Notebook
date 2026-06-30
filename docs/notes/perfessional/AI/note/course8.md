---
comments: true
---

# Cache 系统（Cache Hierarchy & Coherence）

!!! tip "核心要点"
    Cache 的核心设计决策：**Placement**（放哪）、**Replacement**（换谁）、**Write Policy**（怎么写）、**Granularity**（多大块）。多核下还需 **Coherence**（一致性）和 **Consistency**（一致性模型）。

---

## 1. Cache 基础

### 地址分解

给定 Cache 参数，从地址中提取 Tag / Index / Offset：

| 参数 | 位数 |
|------|------|
| Block Offset | $\log_2(block\_size)$ |
| Set Index | $\log_2(\frac{cache\_size}{block\_size \times N\_way})$ |
| Tag | \_bits - index - offset$ |

### 容量术语

| 参数 | 含义 |
|------|------|
| Capacity C | Cache 总数据字节数 |
| Block Size b | 每次装入的数据字节数 |
| Blocks B | C/b |
| Associativity N | 每组的块数 |
| Sets S | B/N |

---

## 2. 三种组织方式

| | Direct-Mapped | Set-Associative | Fully-Associative |
|--|--------------|----------------|-------------------|
| N-way | 1 | 1 < N < B | B |
| Sets | B | B/N | 1 |
| 实现 | 最简单 | **最佳折中** | 最复杂 |
| Conflict | 最高 | 可接受 | 无 |
| 适用 | 简单设计 | **通用** | 小 Cache |

!!! warning "Ping-Pong Effect"
    直接映射中，若 A 和 B 映射到同一 index，交替访问 A,B,A,B,... → **0% hit rate**（每次都是 conflict miss）。

---

## 3. Cache Miss 三种类型

| 类型 | 又叫 | 成因 | 解决 |
|------|------|------|------|
| **Compulsory** | 冷启动 | 首次访问必然 miss | Prefetching |
| **Capacity** | 容量 | Cache 装不下工作集 | 增大 Cache |
| **Conflict** | 冲突 | 多块挤同一位置 | 提高关联度 |

---

## 4. 替换策略

1. 优先换 **Invalid** block
2. Then：
   - **Random**：简单，thrashing 场景优于 LRU
   - **FIFO**：先进先出
   - **LRU**：换最久未用的 → 实现复杂（需记录访问顺序）
   - **Pseudo-LRU**：近似 LRU（如 8-way 用 7 bits 二叉树）
   - **Belady OPT**：换未来最远访问的 → 理论上最优但**无法实现**
   - **Hybrid**：Intel 实际用 LRU+Random 混合

!!! tip "Set Thrashing"
    当工作集大于关联度时，LRU 可能导致 0% hit。此时 **Random 更好**。因此实际用混合策略。

---

## 5. 写策略

| 层次 | 策略 | 行为 |
|------|------|------|
| Store → Cache | **Write-Allocate** | Miss 时先加载到 Cache |
| | Write-No-Allocate | Miss 时直接写内存，绕过 Cache |
| Cache → Memory | **Write-Back** | 只写 Cache, evict 时写回 |
| | Write-Through | 每次同时写 Cache 和内存 |

- Write-Back + Write-Allocate：默认组合，性能最优
- Write-Through + Write-No-Allocate：PCIe/IO/Streaming 场景

---

## 6. 多级 Cache

| 级别 | 特点 |
|------|------|
| L1 | 小、低关联度、延迟最关键、Tag+Data 并行访问、**I/D 分离** |
| L2/L3 | 大、高关联度、延迟次要、Tag+Data 串行访问、**I/D 统一** |

**实际案例**：
- AMD Zen 3 (Ryzen 5000): L1=32KB/core, L2=512KB/core, L3=32MB shared
- IBM POWER10: L2=2MB/core, L3=120MB shared
- NVIDIA A100: L1/Scratchpad=192KB/SM, L2=40MB shared

---

## 7. Cache Coherence（缓存一致性）

### Coherence vs Consistency

| | Coherence | Consistency |
|--|-----------|-------------|
| 范围 | **同一内存地址** | **所有内存地址** |
| 定义 | 各核看到同一地址的最后写入一致 | 所有内存操作的全局顺序约定 |
| 性质 | 局部顺序 | 全局顺序 |

### Coherence 三个性质

1. **Program Order Preservation**：核 C 写地址 X 后读 X，一定读到刚写的值
2. **Coherent Memory View**：核 C1 写 X=1 后，足够长时间后 C2 读到 1
3. **Write Serialization**：对同一地址的写入被所有核以相同顺序看到

### MESI 协议

| 状态 | 含义 | 本地读写行为 |
|------|------|-------------|
| **M**(odified) | 独有 + 已改 | 直接读写，**无需总线** |
| **E**(xclusive) | 独有 + 干净 | 读无需总线；写 → M 也**无需总线** |
| **S**(hared) | 多核共享 | 读无需总线；写需 **Invalidate** 广播 |
| **I**(nvalid) | 不在 Cache | 读/写需总线请求 |

**MESI vs MSI**：新增 E 状态，独占 clean block 时写操作省一次总线广播。

### 实现方式

| | Snoop（嗅探） | Directory（目录） |
|--|-------------|-----------------|
| 机制 | 所有 Cache 监听共享总线 | 目录记录每个 block 的 sharer |
| 扩展性 | 差（总线带宽瓶颈） | 好（点对点通信） |
| 适用 | 小规模多核 | 大规模多核 |

**Directory 条目**（每个 cache line）：
- 2-bit state（M/E/S/I）
- log₂N-bit owner
- N-bit sharer list（one-hot）

---

## 8. Memory Consistency（内存一致性模型）

| 模型 | Load-Load | Load-Store | Store-Store | Store-Load | 代表 |
|------|-----------|------------|-------------|------------|------|
| **Sequential Consistency** | ✅ | ✅ | ✅ | ✅ | 早期理论 |
| **Total Store Order (TSO)** | ✅ | ✅ | ✅ | ❌ | **x86/64** |
| **Partial Store Order (PSO)** | ✅ | ✅ | ❌ | ❌ | **ARM** |
| Weak Memory Model | ❌ | ❌ | ❌ | ❌ | DEC Alpha |

:arrow_right: **越强的一致性模型 → 编程越简单，但性能越低。** x86 选 TSO 是折中。

### 写缓冲区导致的重排序

`
Core 1:              Core 2:
A = 1                B = 1
if (B == 0)          if (A == 0)
  print "Hello"        print "ZJU"
`

在 PSO/弱模型下，**可能同时打印 "Hello" 和 "ZJU"**！因为 Store 可能被重排序到了 Load 之后。

---

## 9. GPU Cache 一致性

- 每个 SM 有独立 L1 Cache
- L1 写**不会立即反映到共享 L2**
- 需手动用 PTX 指令 flush 到 L2/Global Memory
- GPU Cache Coherence 比 CPU 弱得多
