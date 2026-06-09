---	
comments: true	
---	
	
# 自然语言生成与神经机器翻译	
	
"What I cannot create, I do not understand." — Richard Feynman	
	
!!! tip "核心要点"	
    生成的核心是**解码策略**：如何从 $P(Y|X)$ 中选出最优输出序列。贪心、束搜索、采样各有适用场景。NMT 经历了从 SMT → encoder-decoder → attention → Transformer 的演进。	
	
## 1. 自然语言生成 (NLG)	
	
NLG 指生成新文本，是以下任务的子组件：	
	
- 机器翻译、摘要、对话系统	
- 创意写作、诗歌生成	
- 自由形式问答、图像描述	
	
### RNN 语言模型	
	
$$P(Y) = P(y_1) \cdot P(y_2|y_1) \cdot P(y_3|y_1, y_2) \cdots$$	
	
条件语言模型（如 NMT）：$P(Y|X)$，编码器处理 $X$，解码器生成 $Y$。	
	
### Teacher Forcing	
	
训练时，无论解码器预测什么，都将**真实目标词**喂给下一步。加速收敛，但训练-推理不一致（exposure bias）。	
	
## 2. 解码策略	
	
| 策略 | 做法 | 优点 | 缺点 |	
|:---|:---|:---|:---|	
| **贪心搜索** | 每步选概率最高的词 | 简单快速 | 缺乏回溯，质量差 |	
| **束搜索** | 保留 $k$ 个最优部分序列 | 质量好 | $k$ 太大反而降 BLEU |	
| **采样** | 按概率分布随机生成 | 多样性高 | 可能产生不连贯文本 |	
| **Top-n 采样** | 只从 top-n 个词中采样 | 控制多样性 | 需选合适的 n |	
	
### 束搜索 (Beam Search)	
	
- 每步保留 $k$ 个最高概率的部分序列（hypothesis）	
- $k=1$ 退化为贪心搜索	
- $k$ 过大会导致翻译过短（NMT 中 BLEU 不升反降）	
- 需加入长度归一化：$\text{score} = \frac{\log P(Y|X)}{|Y|^\alpha}$	
	
### Softmax Temperature	
	
不是解码算法，而是控制分布"尖锐度"的技巧：	
	
$$P(y_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$	
	
- $T \to 0$：趋向贪心（确定性强）	
- $T \to \infty$：趋向均匀分布（随机性强）	
	
### 模型集成 (Ensembling)	
	
组合多个模型的预测，平滑各自的特异性误差。	
	
## 3. 评估方法	
	
### 人工评估	
	
最可靠但成本高。Flow & Fluency、Adequacy、Completeness。	
	
### 自动评估	
	
| 指标 | 原理 | 特点 |	
|:---|:---|:---|	
| **BLEU** | n-gram 匹配率 | 最常用，偏短文本 |	
| **ROUGE** | 召回率导向 | 适合摘要 |	
| **METEOR** | 词形/同义词匹配 | 与人工更相关 |	
| **Perplexity** | $\exp(-\frac{1}{N}\sum \log P(w_i|w_{<i}))$ | 不涉及生成，只衡量模型 |	
	
## 4. 神经机器翻译 (NMT)	
	
### SMT → NMT	
	
传统统计机器翻译 (SMT) 需要：翻译模型 + 语言模型 + 对齐模型 → 组件多，需人工特征。	
	
NMT 用**单一神经网络**端到端建模 $P(Y|X)$。	
	
### NMT 架构演进	
	
| 版本 | 特点 |	
|:---|:---|	
| **V1: Encoder-Decoder** | 编码器压缩为单一向量，长句退化 |	
| **V2: + Attention** | 解码器可访问所有编码器状态，不受句长限制 |	
| **V3: + Bi-Encoder** | 双向编码器，每个词同时有前后文信息 |	
| **V4: Deep** | 多层堆叠，残差连接解决梯度问题 |	
| **V5: Parallel** | 下一层可在上一层完成前开始计算 |	
	
### 多语言 NMT	
	
单一模型处理多个语言对，只需在源句前加目标语言标签：	
	
- `<2es> How are you` → `Cómo estás`	
- 惊喜发现：zero-shot 翻译可行（从未见过的语言对也能翻译）	
	
### NMT 的挑战	
	
- 训练一次耗时 2-3 周，模型超 100M 参数	
- 搜索空间 $\approx 10^{32}$，计算量巨大	
- 需要大规模双语平行语料	
	
!!! warning "常见误区"	
    束搜索的 beam size 不是越大越好。在 NMT 中，$k$ 过大反而降低 BLEU，因为模型倾向生成过短的翻译。	
	
!!! danger "考试重点"	
    束搜索原理、BLEU 计算、Teacher Forcing 的优缺点、NMT 各版本演进关键创新。	