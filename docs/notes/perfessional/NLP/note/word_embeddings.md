---	
comments: true	
---	
	
# 词嵌入 (Word Embeddings)	
	
将单词表示为稠密向量，使语义相似的词在向量空间中距离相近。	
	
!!! tip "核心要点"	
    词嵌入的核心思想：**"You shall know a word by the company it keeps"** (Firth, 1957)。通过词的上下文分布来表示词义，从 one-hot 稀疏表示转向稠密分布式表示。	
	
## 1. 词的表示	
	
### One-hot 表示的问题	
	
传统 NLP 将词视为原子符号，用 one-hot 向量表示：	
	
- 维度 = 词汇量 (20K ~ 13M)	
- 向量之间正交，无法计算相似度	
- "motel" 和 "hotel" 的内积 = 0，尽管语义相似	
	
### 分布式表示	
	
用稠密向量表示每个词，使其能预测上下文中的其他词：	
	
$$\text{linguistics} = [0.286,\; 0.792,\; -0.177,\; -0.107,\; 0.109,\; -0.542,\; 0.349,\; 0.271]$$	
	
## 2. Word2Vec (Mikolov et al. 2013)	
	
两种架构训练词向量：	
	
### CBOW (Continuous Bag-of-Words)	
	
用上下文词预测中心词：	
	
- 输入：窗口内上下文词的词向量（取平均）	
- 输出：中心词的概率分布	
- 适合小型语料，训练快	
	
### Skip-gram (SG)	
	
用中心词预测上下文词：	
	
- 输入：中心词的词向量	
- 输出：周围词的概率分布	
- 适合大型语料，对罕见词效果更好	
	
## 3. 训练加速	
	
### 问题：Softmax 计算昂贵	
	
标准 Softmax 需要对整个词汇表求和，计算量太大。	
	
### Hierarchical Softmax	
	
用 Huffman 二叉树表示词汇表，将 $|V|$ 分类变为 $\log|V|$ 次二分类：	
	
$$P(w|w_I) = \prod_{j=1}^{L(w)-1} \sigma\left(v_{n(w,j)}^\top v_{w_I}\right)$$	
	
### Negative Sampling	
	
将多分类简化为二分类：区分真实上下文词和随机采样的噪声词。	
	
$$\log\sigma(v_w^\top v_c) + \sum_{i=1}^k \mathbb{E}_{w_i\sim P_n(w)}\left[\log\sigma(-v_{w_i}^\top v_c)\right]$$	
	
!!! warning "常见误区"	
    Skip-gram + Negative Sampling (SGNS) 实际上是在对 **PMI 矩阵做隐式因式分解**（Levy & Goldberg, 2014），并非纯粹的语言模型。	
	
## 4. 词向量的性质	
	
### 词类比 (Word Analogy)	
	
词向量捕获了语义和语法关系：	
	
$$\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$$	
	
### 可视化	
	
相近含义的词语在向量空间中聚集：apple, orange, banana → 水果簇；car, bus, train → 交通工具簇。	
	
## 5. 词向量的应用	
	
| 应用 | 传统方法 | 词向量改进 |	
|:---|:---|:---|	
| **词相似度** | WordNet, Edit Distance | 利用上下文相似性自动捕获同义词 |	
| **机器翻译** | 基于规则 | 跨语言词向量对齐 |	
| **POS/NER 标注** | MEMM, CRF | 作为特征提升序列标注 |	
| **关系抽取** | OpenIE, Bootstrapping | 稠密特征提升泛化 |	
| **情感分析** | Naive Bayes, SVM | 余弦距离直接比较情感 |	
	
## 6. 局限性	
	
- **一词多义**：每个词只有一个向量，无法区分 "bank"（银行/河岸）	
- **OOV 问题**：无法处理未登录词	
- **序列信息缺失**：词袋模型，忽略词序	
- **可解释性差**：维度不可解释	
	
!!! danger "考试重点"	
    CBOW vs Skip-gram 的区别、Negative Sampling 的原理、词类比公式的含义。	