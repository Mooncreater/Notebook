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