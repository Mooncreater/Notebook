---	
comments: true	
---	
	
# Transformer	
	
2017 年提出的架构，完全基于注意力机制，取代 RNN 成为 NLP 主流。	
	
!!! tip "核心要点"	
    Transformer 的核心是 **Self-Attention**：序列中每个位置都关注所有位置，$O(1)$ 的最大路径长度（RNN 是 $O(n)$）。可以并行计算，训练效率远超 RNN。	
	
## 1. 为什么取代 RNN？	
	
RNN 的痛点：	
	
- **难以并行**：$h_t$ 依赖 $h_{t-1}$，必须逐时间步计算	
- **长距离依赖退化**：即使 LSTM，梯度仍会衰减	
	
CNN 替代方案：多层卷积扩大感受野，可并行。但 Self-Attention 更直接。	
	
## 2. Self-Attention 机制	
	
### 核心计算	
	
每个输入 $x_i$ 通过三个权重矩阵映射：	
	
$$q_i = W^q a_i,\quad k_i = W^k a_i,\quad v_i = W^v a_i$$	
	
- **Query (Q)**：去匹配其他位置	
- **Key (K)**：被 Q 匹配	
- **Value (V)**：实际提取的信息	
	
**缩放点积注意力**：	
	
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$	
	
### 矩阵形式	
	
$$Q = W^q X,\quad K = W^k X,\quad V = W^v X$$	
$$A = QK^\top$$	
$$\hat{A} = \text{softmax}\left(\frac{A}{\sqrt{d_k}}\right)$$	
$$O = \hat{A}V$$	
	
全部为矩阵乘法，GPU 高度可并行！	
	
### 为什么除以 $\sqrt{d_k}$？	
	
当 $d_k$ 很大时，点积值变大，softmax 进入梯度极小区（饱和区）。除以 $\sqrt{d_k}$ 保持方差为 1。	
	
## 3. Multi-Head Attention	
	
单头注意力可能只关注一种关系。多头允许模型同时关注不同子空间：	
	
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$	
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$	
	
## 4. Transformer 完整架构	
	
### 编码器 (Encoder)	
	
每层包含：	
	
1. Multi-Head Self-Attention	
2. Feed-Forward Network (FFN)：$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$	
3. 每个子层后：LayerNorm + 残差连接	
	
$$x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$$	
	
### 解码器 (Decoder)	
	
每层包含：	
	
1. **Masked** Multi-Head Self-Attention（防止看到未来信息）	
2. Cross-Attention（Q 来自解码器，K/V 来自编码器输出）	
3. FFN + 残差 + LayerNorm	
	
### 位置编码 (Positional Encoding)	
	
Self-Attention 没有序列顺序感知。通过正弦位置编码注入位置信息：	
	
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$	
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$	
	
## 5. 预训练范式	
	
### BERT：掩码语言模型 (MLM)	
	
- 随机 mask 15% 的 token，用上下文预测被 mask 的词	
- 双向 Transformer 编码器	
- 输入格式：`[CLS] Sentence [SEP]`	
	
### GPT：自回归语言模型	
	
- 预测下一个 token（单向/因果注意力）	
- Transformer 解码器	
- 通过 masked self-attention 保证只看到过去信息	
	
!!! warning "常见误区"	
    BERT 用的是 Transformer **编码器**（双向），GPT 用的是 Transformer **解码器**（单向）。两者预训练任务不同：BERT 做"完形填空"，GPT 做"续写"。	
	
### BERT 的微调策略	
	
| 任务类型 | 做法 |	
|:---|:---|	
| 单句分类 | 取 `[CLS]` 输出 + 分类头 |	
| 序列标注 | 每个 token 输出 + 分类头 |	
| 句子对 | `[CLS]` S1 `[SEP]` S2 `[SEP]` |	
| 抽取式 QA | 预测答案的 start/end 位置 |	
| 序列生成 | Encoder-Decoder (如 BART) |	
	
### Adaptor 微调	
	
在预训练模型层间插入小模块，只训练 adaptor + task head，冻结预训练参数。参数效率高，多任务友好。	
	
## 6. 预训练方法	
	
### 自监督学习	
	
不需要人工标注，模型从数据自身创建监督信号：	
	
| 方法 | 预训练任务 | 代表模型 |	
|:---|:---|:---|	
| MLM | 预测被 mask 的 token | BERT |	
| NSP/句子顺序 | 判断两句话是否相邻 | BERT |	
| 自回归 LM | 预测下一个 token | GPT, GPT-2, GPT-3 |	
| 双向 LM | 前向 + 后向 LM 组合 | ELMo |	
	
### 为什么预训练有效？	
	
海量无标注数据 → 学习通用语言知识 → 少量标注数据即可适配下游任务。	
	
!!! danger "考试重点"	
    Self-Attention 公式推导、Multi-Head Attention 原理、BERT vs GPT 架构差异、位置编码的意义、残差连接 + LayerNorm 的作用。	