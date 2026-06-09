---	
comments: true	
---	
	
# 循环神经网络 (RNN)	
	
RNN 是一类具有"记忆"的神经网络，专门处理序列数据，在 NLP 中被广泛使用。	
	
!!! tip "核心要点"	
    RNN 的核心是一组**共享参数** $f$，无论输入序列多长，都只用一个函数。$h_t = f(h_{t-1}, x_t)$，隐藏状态 $h$ 携带历史信息。	
	
## 1. 基本 RNN	
	
### 结构	
	
$$h_t = \sigma(W_h h_{t-1} + W_x x_t)$$	
$$y_t = \text{softmax}(W_o h_t)$$	
	
- $h_t$：时刻 $t$ 的隐藏状态	
- $x_t$：时刻 $t$ 的输入	
- $W_h, W_x, W_o$：共享权重矩阵	
	
### 变体	
	
- **Deep RNN**：多层 RNN 堆叠，每层有自己的隐藏状态	
- **双向 RNN (BiRNN)**：同时从前后两个方向读取序列，捕获完整上下文	
	
## 2. LSTM (长短期记忆网络)	
	
### 为何需要 LSTM？	
	
朴素 RNN 存在**梯度消失/爆炸**问题，难以捕获长距离依赖。	
	
### 核心思想	
	
LSTM 引入**细胞状态** (cell state) $c_t$，通过门控机制控制信息流动：	
	
- $c_t$ 变化缓慢，携带长期记忆	
- $h_t$ 变化较快，反映当前输出	
	
### 三个门控	
	
给定输入 $x_t$ 和前一隐藏状态 $h_{t-1}$：	
	
| 门 | 符号 | 作用 |	
|:---|:---|:---|	
| **遗忘门 (Forget Gate)** | $f_t = \sigma(W_f[h_{t-1}, x_t])$ | 决定丢弃哪些旧信息 |	
| **输入门 (Input Gate)** | $i_t = \sigma(W_i[h_{t-1}, x_t])$ | 决定存储哪些新信息 |	
| **输出门 (Output Gate)** | $o_t = \sigma(W_o[h_{t-1}, x_t])$ | 决定输出哪些信息 |	
	
**状态更新**：	
	
$$c_t = f_t \odot c_{t-1} + i_t \odot \tanh(W_c[h_{t-1}, x_t])$$	
$$h_t = o_t \odot \tanh(c_t)$$	
	
!!! warning "常见误区"	
    Forget gate 对 LSTM 性能**至关重要**，输出门的激活函数也很关键。将输入门与遗忘门耦合（即 $i_t = 1 - f_t$）是一种常用简化。	
	
## 3. GRU (门控循环单元)	
	
GRU 是 LSTM 的简化版，只有两个门：	
	
- **重置门 (Reset Gate)** $r_t$：控制忽略多少过去信息	
- **更新门 (Update Gate)** $z_t$：控制保留多少过去信息（类似 LSTM 的 forget + input）	
	
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$	
	
参数更少，训练更快，效果接近 LSTM。	
	
## 4. RNN 在 NLP 中的应用	
	
NLP 充满序列数据，RNN 擅长捕获**长距离依赖**：	
	
- **主谓一致**：He does not have ... himself. / She does not have ... herself.	
- **指代消解**："The trophy would not fit in the brown suitcase because **it** was too big."	
- **选择偏好**：reign → queen / rain → clouds	
	
### 应用场景	
	
| 模式 | 应用 |	
|:---|:---|	
| 读取整个句子 → 预测标签 | 句子分类、条件生成、检索 |	
| 读取上下文 → 逐词预测 | 序列标注、语言模型、依存解析 |	
	
## 5. 编码器-解码器 (Encoder-Decoder)	
	
用于序列到序列任务（如机器翻译）。	
	
- **编码器**：将输入序列编码为上下文向量	
- **解码器**：从上下文向量生成输出序列	
	
## 6. 注意力机制 (Attention)	
	
### 动机	
	
编码器-解码器的瓶颈：所有输入信息被压缩到单一向量中。	
	
### 计算步骤	
	
1. 编码器为每个输入词生成 key-value 对	
2. 解码器每步生成 query，与所有 key 计算注意力权重	
3. 用权重对 value 加权求和，得到上下文向量	
	
$$\alpha_i = \text{softmax}(\text{score}(q, k_i))$$	
$$c = \sum_i \alpha_i v_i$$	
	
### 注意力打分函数	
	
- 点积：$q^\top k$	
- 缩放点积：$q^\top k / \sqrt{d}$	
- 加性：$v^\top \tanh(W[q; k])$	
	
### 变体	
	
- **Hard Attention**：离散选择关注位置（需强化学习训练）	
- **Self-Attention**：序列内部元素互相关注 → 这是 Transformer 的基础	
- **Multi-head Attention**：多组并行的注意力	
- **层级注意力**：词级 → 句级 → 文档级	
	
## 7. Pointer Network	
	
输出指向输入序列中的位置，而非从固定词汇表生成。	
	
- 应用：摘要（从原文复制）、机器翻译（处理罕见词）	
	
!!! danger "考试重点"	
    LSTM 三个门的作用和公式、Attention 的计算流程、Self-Attention 的含义是核心考点。	