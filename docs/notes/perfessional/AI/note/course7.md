---
comments: true
---

# GPU 优化（GPU Optimization）

!!! tip "核心要点"
    GPU 优化的四大支柱：**多线程隐藏延迟**（Occupancy）、**合并访存**（Memory Coalescing）、**共享内存**（Shared Memory）、**避免 Warp Divergence**。外加 CPU-GPU 数据传输优化。

---

## 1. 延迟隐藏与 Occupancy

### FGMT（Fine-Grained Multi-Threading）

当一个 Warp 等待内存访问时，SM 立即切换到另一个就绪的 Warp → 隐藏长延迟。

**Occupancy（占用率）**：
Occupancy = \frac{Active\ Warps}{Max\ Warps\ per\ SM}

- 越高越好：更多 Warp 可切换 → 更能隐藏延迟
- 受限于：寄存器数量、Shared Memory 大小、Block 大小

### ILP（Instruction Level Parallelism）

单个 Warp 内多条指令可重叠执行：
- 例如：Load Unit + Multiply Unit + Add Unit 同时工作
- 1 warp/cycle 发射，但完成 24 ops/cycle（8 lanes × 3 units）

---

## 2. 内存合并（Memory Coalescing）

**问题**：同一 Warp 内线程访问的全局内存地址如果**不连续**，会退化为多次内存事务。

**优化**：让相邻线程访问相邻地址。

`
// 不合并（Strided）：Thread i → A[i*N]
// 合并（Coalesced）：Thread i → A[tid]
`

同一 Warp（32 threads）如果访问一个连续的 128-byte 对齐区域 → **1 次内存事务**（而非 32 次）。

---

## 3. 共享内存（Shared Memory）

**位置**：每个 SM 内部，~几十 KB

**特点**：
- 延迟 ~5 cycles（vs Global Memory ~500 cycles）
- 同一 Block 内所有线程共享
- **手动管理**：程序员显式加载/存储

**Tiled Matrix Multiplication 经典用法**：

`cuda
__shared__ float A_s[TILE][TILE];
__shared__ float B_s[TILE][TILE];

for (tile = 0; tile < N/TILE; ++tile) {
    // 从 Global Memory 加载 tile 到 Shared Memory
    A_s[threadIdx.y][threadIdx.x] = A[row*N + tile*TILE + threadIdx.x];
    B_s[threadIdx.y][threadIdx.x] = B[(tile*TILE+threadIdx.y)*N + col];
    __syncthreads();  // 等待所有线程加载完毕
    
    // 用 Shared Memory 中的数据计算
    for (i = 0; i < TILE; ++i)
        sum += A_s[threadIdx.y][i] * B_s[i][threadIdx.x];
    __syncthreads();  // 等待所有线程计算完毕再加载下一 tile
}
`

**Shared Memory 的问题**：
- 手动管理：程序员痛苦
- CPU 上的 Cache 是**自动管理**的（硬件透明），GPU 需要显式搬数据

---

## 4. Warp Divergence（线程束分化）

同一 Warp 内线程走不同分支路径 → **串行执行**各路径，屏蔽不活跃线程。

`cuda
if (tid % 2 == 0) {
    // Path A: 偶数线程
} else {
    // Path B: 奇数线程
}
`

:arrow_right: Warp 先执行 Path A（奇数线程屏蔽），再执行 Path B（偶数线程屏蔽）→ **2x 时间**

**优化**：让同一 Warp 内的线程走相同分支路径。

---

## 5. CPU-GPU 数据传输优化

**瓶颈**：PCIe 带宽远小于 GPU 显存带宽

**优化手段**：
- **减少传输**：中间数据留在 GPU
- **异步传输**：cudaMemcpyAsync + Stream，与计算重叠
- **固定内存**（Pinned Memory）：比 pageable memory 传输快
- **批量传输**：一次传大数据块，避免多次小传输

---

## 6. Atomic 操作

多线程同时更新同一内存位置时使用 tomicAdd 等。

**问题**：同一 Warp 内多线程对同一地址做 atomic → **串行化**，性能差

**优化**：先用 Shared Memory 做 Block 内归约，再 atomic 到 Global Memory

---

## 7. GPU 优化速查表

| 优化方向 | 方法 | 收益 |
|----------|------|------|
| 延迟隐藏 | 提高 Occupancy | 隐藏 Global Memory 延迟 |
| 访存效率 | Memory Coalescing | 减少内存事务数 |
| 数据复用 | Shared Memory | ~100x 延迟降低 |
| 分支效率 | 避免 Warp Divergence | 避免串行化 |
| 传输效率 | Pinned Memory + 异步 | 隐藏 PCIe 延迟 |
| Bank Conflict | 避免 Shared Memory bank 冲突 | 提高 Shared Memory 吞吐 |
