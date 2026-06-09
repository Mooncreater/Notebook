---	
comments: true	
---	
	
# 模型压缩与高效微调	
	
大模型推理比训练更贵。三大压缩技术：量化、剪枝、蒸馏。参数高效微调让下游适配只需极少参数。	
	
!!! tip "核心要点"	
    模型压缩三件套：**量化**（降精度）、**剪枝**（去冗余）、**蒸馏**（小模型学大模型）。高效微调让每个下游任务只训练极少量参数（Adapter、LoRA、Prompt Tuning）。	
	
## 1. 模型压缩 (Model Compression)	
	
### 为什么需要压缩？	
	
- 顶尖模型参数量巨大（Llama 2, GPT-3 等）	
- 推理成本比训练更高（每查询都要跑完整前向）	
- 部署在边缘设备需要小模型	
- 过参数化理论：过参数化模型反而更容易优化 (Du & Lee 2018)	
	
### 三种基本方法	
	
| 方法 | 做法 | 核心思想 |	
|:---|:---|:---|	
| **量化 (Quantization)** | 减少每参数的比特数 | 保持结构，降精度 |	
| **剪枝 (Pruning)** | 移除不重要的参数 | 去除冗余连接 |	
| **蒸馏 (Distillation)** | 小模型模仿大模型 | 知识迁移 |	
	
## 2. 量化 (Quantization)	
	
### 浮点数表示	
	
$$v = (-1)^s \cdot M \cdot 2^E$$	
	
- $s$：符号位	
- $M$：尾数（小数部分）	
- $E$：指数部分	
	
### Absmax 量化 (INT8)	
	
将 FP32 值映射到 [-127, 127]：	
	
$$x_q = \text{round}\left(\frac{127}{\max|x|} \cdot x\right)$$	
	
例：`[0.5, 20, -0.0001, -0.01, -0.1]` → `[3, 127, 0, 0, -1]`	
	
### 模型感知量化 (GOBO, LLM.int8)	
	
- **GOBO**：BERT 权重呈高斯分布，99.9% 权重量化为 8 桶，保留异常值 0.01%	
- **LLM.int8**：Transformer 中 95% 是矩阵乘法，对该操作进行混合精度分解	
	
### Binarized Neural Networks (BNN)	
	
极端量化：权重和激活值都是二值（-1 / 1）。	
	
### 量化感知训练 (QAT)	
	
在训练时就模拟量化效果，比训练后量化 (PTQ) 精度损失更小。	
	
### QLoRA (Dettmers et al. 2023)	
	
4-bit 量化 + LoRA = 可在 48GB GPU 上训练 65B 模型！	
	
## 3. 剪枝 (Pruning)	
	
### 量化 vs 剪枝	
	
| | 量化 | 剪枝 |	
|:---|:---|:---|	
| 操作 | 降低参数精度 | 将参数置零 |	
| 参数数量 | 不变 | 减少 |	
	
### 幅度剪枝 (Magnitude Pruning)	
	
将绝对值最小的 X% 参数置零。属于**非结构化剪枝**。	
	
### Lottery Ticket Hypothesis (Frankle et al. 2018)	
	
随机初始化的网络中"藏有"一个子网络，单独训练可以达到甚至超过完整网络的性能。	
	
### Wanda (Sun et al. 2024)	
	
基于权重幅度和激活值进行剪枝，无需重新训练。	
	
## 4. 蒸馏 (Knowledge Distillation)	
	
用大模型（Teacher）的输出来训练小模型（Student）。	
	
### Soft Targets (Hinton et al. 2015)	
	
不仅用 hard label，还用 Teacher 的概率分布作为软目标：	
	
$$\mathcal{L}_{KD} = \text{KL}\left(\text{softmax}\left(\frac{z_T}{T}\right), \text{softmax}\left(\frac{z_S}{T}\right)\right)$$	
	
- $T$：温度参数，$T>1$ 使分布更平滑，暴露类间关系	
	
### Sequence-Level Distillation (Kim & Rush 2016)	
	
- Word-level：每步匹配 Teacher 的词分布	
- Sequence-level：最大化 Teacher 生成的整个输出序列	
	
### DistilBERT (Sanh et al. 2019)	
	
- 一半层数，60% 参数	
- 用 BERT 交替层初始化	
- 同时使用监督损失和蒸馏损失 + 隐藏状态余弦相似度	
	
### Self-Instruct (Wang et al. 2022)	
	
用模型自身生成指令并伪标注，训练指令跟随能力。	
	
## 5. 参数高效微调 (PEFT)	
	
### 动机	
	
- 全量微调每个下游任务要保存一份完整模型（BERT 110M × N个任务）	
- 数据稀缺 + 模型持续变大	
	
### 方法对比	
	
| 方法 | 做法 | 可训练参数量 |	
|:---|:---|:---|	
| **Adapter** | 在层间插入小瓶颈模块 | ~1-3% |	
| **LoRA** | 低秩分解模拟权重更新 | ~0.1-1% |	
| **Prefix Tuning** | 在输入前加可训练前缀向量 | <1% |	
| **Prompt Tuning** | 学习软提示向量 | <0.1% |	
| **BitFit** | 只微调 bias 项 | ~0.1% |	
	
### PEFT 的优势	
	
- **存储高效**：每个任务只需保存少量参数	
- **抗过拟合**：在少样本、OOD 场景表现更好	
- **多任务友好**：一个主干 + 多个 adapter	
	
## 6. 训练策略分类	
	
| 策略 | 数据 | 参数 |	
|:---|:---|:---|	
| **Zero-shot** | 0 样本 | 不更新 |	
| **Few-shot** | 1-100 样本 | 可能更新 |	
| **Full-data Fine-tuning** | 10K+ 样本 | 更新 |	
	
### PET (Pattern-Exploiting Training)	
	
半监督方法：	
	
1. 用不同 prompt 模板 + 标注数据训练多个 PLM	
2. 对无标注数据打软标签	
3. 用软标签数据训练最终分类器	
	
### Early Exit	
	
在 Transformer 每层加分类头，当置信度足够高时提前退出，减少推理时间。	
	
!!! danger "考试重点"	
    量化 vs 剪枝 vs 蒸馏三种压缩方法的区别、LoRA 的原理、Adapter 的结构、PET 三步骤。	