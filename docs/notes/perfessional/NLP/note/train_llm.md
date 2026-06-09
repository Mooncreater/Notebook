---	
comments: true	
---	
	
# 如何训练大语言模型	
	
LLM 太大，单 GPU 装不下。需要分布式训练、内存优化和量化等工程技术。	
	
!!! tip "核心要点"	
    8B 模型的训练需要处理三大内存挑战：**参数+梯度+优化器状态**（用 ZeRO 分片）、**激活值**（用 Flash Attention）、**精度**（用量化）。梯度累积让小 batch 模拟大批量。	
	
## 1. LLM 训练的内存挑战	
	
典型的 PyTorch 训练循环在 LLM 上直接跑不了：	
	
```python	
for epoch in range(num_epochs):	
    for batch in dataloader:	
        outputs = model(inputs)	
        loss = loss_fn(outputs, targets)	
        optimizer.zero_grad()	
        loss.backward()	
        optimizer.step()	
```	
	
原因：8B 模型参数就超单 GPU 显存。	
	
### 三大内存占用	
	
| 类别 | 内容 | 大致占比 |	
|:---|:---|:---|	
| **Part I** | 模型参数 + 梯度 + 优化器状态 | ~70-80% |	
| **Part II** | 激活值 (activations) | ~15-25% |	
| **Part III** | 量化精度 | 可选压缩 |	
	
### Batch Size 要求	
	
需要足够大的 batch 提供清晰梯度信号。一般 4M-60M tokens / batch。	
	
- DeepSeek V3：batch size 1920 × 32K context ≈ 61M tokens	
- **梯度累积**：`Global BS = Mini BS × Gradient Accumulation Steps`，如 1920 = 16 × 120	
	
## 2. Part I：参数、梯度、优化器状态	
	
### 多 GPU 策略	
	
使用 DeepSpeed 的 **ZeRO (Zero Redundancy Optimizer)**：	
	
| 阶段 | 分片内容 | 内存节省 |	
|:---|:---|:---|	
| ZeRO-1 | 优化器状态 | ×4 |	
| ZeRO-2 | + 梯度 | ×8 |	
| ZeRO-3 | + 模型参数 | ×N (N=GPU数) |	
	
### ZeRO-Offload	
	
将优化器状态和梯度卸载到 CPU 内存（但速度显著变慢）。	
	
## 3. Part II：激活值优化	
	
### Flash Attention	
	
核心思想：不将完整注意力矩阵写回 HBM（高带宽显存），而是分块计算，减少内存读写。	
	
- 标准 Self-Attention：$O(N^2)$ 内存	
- Flash Attention：通过 tiling 和 recomputation，内存降低至 $O(N)$	
	
### Liger Kernel	
	
用优化后的 Triton 代码重写核心计算，一行代码即可启用：	
	
```python	
from liger_kernel.transformers import AutoLigerKernelForCausalLM	
model = AutoLigerKernelForCausalLM.from_pretrained("path/to/model")	
```	
	
## 4. Part III：量化	
	
有损压缩，用低精度表示模型参数以节省显存：	
	
- **FP16/BF16**：半精度训练，最常用	
- **INT8**：8 位整数量化	
- **INT4**：进一步压缩（QLoRA 用 4-bit 训 65B 模型于 48GB GPU）	
	
### 训练 vs 推理量化	
	
- 训练时：混合精度（FP16 前向 + FP32 权重备份）	
- 推理时：可进一步量化（INT8/INT4）	
	
## 5. 分布式训练并行策略	
	
| 策略 | 做法 | 适用场景 |	
|:---|:---|:---|	
| **数据并行 (DP)** | 每 GPU 拷贝模型，分数据 | 模型小、数据大 |	
| **模型并行 (MP)** | 模型切分到多 GPU | 模型太大单卡装不下 |	
| **流水线并行 (PP)** | 按层切分，流水线执行 | 结合 MP 使用 |	
| **张量并行 (TP)** | 单层内矩阵切分 | Megatron-LM |	
| **混合并行** | 组合以上策略 | 大规模训练（如盘古） |	
	
!!! danger "考试重点"	
    ZeRO 三个阶段的区别、Flash Attention 为什么省内存、梯度累积的原理和公式。	