---
comments: true
---

# 昇腾（Ascend DaVinci）+ TPU + Cambricon

!!! tip "核心要点"
    AI 芯片三巨头：**Google TPU**（脉动阵列，生态好）、**华为昇腾 DaVinci**（高能效，Cube 算力极致）、**寒武纪 Cambricon**（高可编程性，DLA 架构）。每个都是"专用数据通路 + 片上存储层次 + 领域指令集"。

---

## 1. 华为昇腾 DaVinci

### 1.1 AI Core 三大模块

| 模块 | 角色 | 功能 |
|------|------|------|
| **Cube Unit** | 算力担当 | 一拍完成 fp16 的 2 个 16×16 矩阵乘 |
| **Vector Unit** | 多面手 | 向量运算（激活、池化等），SIMD 长度 128(fp16)/64(fp32) |
| **Scalar Unit** | 司令部 | 标量运算、循环控制、分支判断、地址计算 |

### 1.2 Cube 模块详解

- 矩阵乘：C = A × B，一拍完成 2 个 16×16 fp16 矩阵乘
- int8：一拍完成 16×32 × 32×16
- Accumulator：C = A × B + C（支持卷积 bias 加法）
- L0A/L0B/L0C Buffer：片上专用缓存
- A/B DFF + Accum DFF：数据寄存器

### 1.3 Vector 模块详解

- 覆盖 FP16/FP32/int32/int8 等数据类型
- 支持连续或固定间隔寻址（Stride）
- Unified Buffer (UB)：Vector 运算的源和目的
- 数据搬运附带：ReLU / 格式转换（**随路计算**）

### 1.4 Scalar 模块详解

- 功能类似小 CPU
- GPR：32 个通用寄存器
- SPR：专用寄存器（CoreID, BLOCKID, VA, STATUS, CTRL...）
- 不能直接访问外部 DDR/HBM → 用 UB 或 Scalar Buffer

### 1.5 DaVinci 优劣势

| 优势 | 不足 |
|------|------|
| **Cube 极致算力高**（同等面积功耗下算力是 V100 的 2.1 倍） | 难编程（事件同步、Buffer 管理） |
| Buffer 访问管理效率高 | 生态不完善（Debug、PMU 工具少） |
| 硬件随路计算指令（Img2Col、格式转换） | — |

---

## 2. Google TPU

### 2.1 各代演进

| 版本 | 用途 | 关键特性 |
|------|------|----------|
| **TPU v1** | Inference only | 256×256 MAC Systolic Array, DDR3, PCIe |
| **TPU v2** | Training + Inference | — |
| **TPU v3** | Training + Inference | 更强算力 |
| **TPU v4** | Training | TPU4i for Inference |

### 2.2 TPU v1 架构

- **Matrix Multiply Unit**：256×256 MAC（8-bit），Systolic Array，占 24% 面积
- **Unified Buffer**：24 MB，占 29% 面积
- 模型预存在 DDR3，数据从 Host 通过 PCIe 来
- 仅推理

### 2.3 Systolic Array 数据流

`
PE 更新规则：Right = Left; Down = Upper; Cell += Upper × Left

左矩阵 → 从左流入，向右传播
右矩阵 → 从上流入，向下传播
结果   → 在 PE 内累加，完成后流出
`

---

## 3. 寒武纪 Cambricon

### 3.1 设计目标

解决两大核心问题：
1. **提高性能/功耗比**（Performance/Power ratio）
2. **提高可编程性**（Programmability）

### 3.2 DLP-S（单核深度学习处理器）

| 模块 | 组件 | 功能 |
|------|------|------|
| **Control** | IFU, IDU | 取指、译码、发射 |
| **Compute** | VFU (向量), MFU (矩阵) | 张量级运算 |
| **SRAM** | WRAM (权重), NRAM (神经元) | 片上存储，分类管理 |

### 3.3 执行流程（7 步）

`
Step 1: IFU 取指令 → IDU 译码 → 分发到 DMA/VFU/MFU
Step 2: DMA 从 DRAM 读 Neuron Tensor → NRAM, Weight Tensor → WRAM
Step 3: VFU 从 NRAM 读神经元 tensor → 预处理（边界扩充等） → 发给 MFU
Step 4: MFU 从 VFU 接收神经元 + 从 WRAM 读权重 → 矩阵运算 → 结果发 VFU
Step 5: VFU 后处理（激活、池化等）
Step 6: VFU 写结果回 NRAM
Step 7: DMA 写 NRAM → DRAM
`

### 3.4 DLP-M（多核处理器）

- 分层结构：DLP-M → 多个 DLP-C → 多个 DLP-S
- 减少 NoC（Network-on-Chip）负载和开销

### 3.5 指令集分类

| 类别 | 示例指令 | 说明 |
|------|----------|------|
| **Control** | JUMP, CB | 跳转、条件分支 |
| **Data Movement** | MLOAD/MSTORE, VLOAD/VSTORE, MOVE | 主存↔片上、片上传输 |
| **Compute** | MMV, VMM, VAV, IP, VEXP, VLOG | 矩阵/向量/标量运算 |
| **Logic** | VGT, VE, VAND, VOR, VNOT, VGTM | 向量/标量逻辑 |

### 3.6 指令发射

- 三个 Issue Queue：Control IQ, Compute IQ, Memory IQ
- **Queue 之间**：乱序执行（Out of Order）
- **Queue 内部**：顺序执行（In Order）
- 队列间插入 SYNC 指令同步

---

## 4. 三款芯片对比

| | Google TPU | 华为昇腾 | 寒武纪 |
|--|-----------|----------|--------|
| 计算核心 | Systolic Array 256×256 | Cube (16×16×2) + Vector | MFU(矩阵) + VFU(向量) |
| 存储 | Unified Buffer 24MB | L0 + UB | WRAM + NRAM |
| 编程 | TensorFlow | CANN | BANG C |
| 强项 | 生态成熟 | 能效极致 | 可编程性高 |
| 弱项 | 非开源 | 生态起步 | — |
