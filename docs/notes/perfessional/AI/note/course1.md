---	
comments: true	
---	
	
# 性能基础	
	
!!! tip "核心要点"	
    Amdahl 定律告诉我们并行的上限由串行部分决定。Roofline 模型帮我们判断瓶颈在计算还是访存。	
	
## Amdahl's Law（阿姆达尔定律）	
	
描述并行化的加速上限。程序中有一个固定串行部分，限制了最大加速比。	
	
- $f$：可并行部分占总任务的比例	
- $n$：处理器数量	
	

$$Speedup = \frac{1}{(1 - f) + \frac{f}{n}}$$	
	
当 $n \to \infty$ 时：$Speedup_{max} = \frac{1}{1 - f}$	
	
:arrow_right: 即使处理器无限多，串行部分仍是瓶颈。	
	
!!! danger "考试重点"	
    极限加速比公式 $Speedup_{max} = \frac{1}{1-f}$ 经常考。若 90% 可并行（$f=0.9$），最多加速 10 倍，再多核也白搭。	
	
## Roofline Model（屋顶线模型）	
	
分析计算机性能的模型，评估计算平台的**算力上限**与实际能达到的性能。	
	
### 核心指标	
	
**运算强度（Operational Intensity）**：	
	

$$OI = \frac{FLOPs}{Bytes} = \frac{\text{计算量}}{\text{访存量}}$$	
	
单位：FLOPS/Byte —— 每字节数据做多少次浮点运算。	
	
**可达到的性能**：	
	

$$Performance = \min(Peak\_Compute, \; Peak\_Bandwidth \times OI)$$	
	
### 两个区域	
	
- **Compute Bound（计算瓶颈）**：OI 高，受算力限制（屋顶平坦部分）	
- **Memory Bound（访存瓶颈）**：OI 低，受带宽限制（屋顶斜坡部分）	
	
## Little's Law（利特尔法则）	
	
队列系统中并发量、吞吐率与延迟的关系：	
	

$$L = \lambda \times W$$	
	
- $L$：系统中平均任务数	
- $\lambda$：平均到达率（吞吐率）	
- $W$：平均完成时间（延迟）	
	
应用：若已知 IPC 和每条指令延迟，可推算流水线需要的缓冲大小。	
	
## 性能度量	
	
| 指标 | 含义 |	
|------|------|	
| **CPI** | Cycles Per Instruction —— 每条指令所需时钟周期，越低越好 |	
| **IPC** | Instructions Per Cycle —— 每周期执行指令数，CPI 的倒数 |	
| **MIPS** | Million Instructions Per Second |	
| **FLOPS** | Floating-Point Operations Per Second |

### Memory Roofline 变体

不同存储层级有不同的带宽上限 → 不同"屋顶"：

| 存储层级 | 带宽 | 屋顶形状 |
|----------|------|----------|
| DRAM | 有限 (~100 GB/s) | 低斜坡 |
| HBM | 中等 (~1 TB/s) | 中斜坡 |
| Cache | 大 (~10 TB/s) | 高斜坡 |

应用的实际 OI 在哪一层，就受那一层带宽限制。

### 经典 Roofline 解题步骤

1. 计算 OI = FLOPs / Bytes
2. 读图：找到 OI 在 X 轴的位置
3. 向上作垂直线，与屋顶的交点 = 可达到性能
4. 若交点在斜坡上 → Memory Bound（优化：减少访存）
5. 若交点在平顶上 → Compute Bound（优化：更好算法/硬件）

**示例**：Peak=100 GFLOPS, BW=25 GB/s, OI=2 FLOP/Byte
Perf = \min(100, 25 \times 2) = 50\ GFLOPS
:arrow_right: 仅达峰值 50%，Memory Bound，需优化访存！

### Roofline 的作用

1. **从处理器角度**：展示硬件固有上限（计算/内存）
2. **从计算核角度**：指示优化优先级
3. **从应用角度**：评估当前瓶颈是延迟还是吞吐
