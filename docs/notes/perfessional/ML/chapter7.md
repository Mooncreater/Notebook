## 第七章
<h2 style="color: #2931d9ff; font-weight: normal;"> 贝叶斯分类器</h2>

通过训练集概率分布预测 $\bm{x}$ 应该被分类到何处，建模 $P(c|\bm{x})$.

#### 1.基本理论
将样本 $\bm{x}$ 分到第 $i$ 类的条件风险：$R(c_i|\bm{x}) = \sum \limits_{j=1}^{N}\lambda_{ij}P(c_j|\bm{x})$
判定准则：$h^*(\bm{x}) = \argmin \limits_{c\in \mathcal{Y}}R(c |\bm{x})$. <span style = "background-color : #00bbffff">  选择风险最小的分类 </span>

策略？
- [ ]  判别式
- [x]  生成式

#### 2.贝叶斯定理
如何得到 $P(c | \bm{x})$ ？

- [x] 已知 $P(c)$
- [x] 求 $P(\bm{x} |c)$
- [x] 使用贝叶斯定理: $P(c|\bm{x}) = \frac{P(c)P(\bm{x}|c)}{P(\bm{x})}$

#### 3.极大似然估计

##### 3.1 什么是似然？

我们看到了数据（结果）。
我们猜测是某个参数（原因）导致了它。
我们评估这个原因（参数值）的“像然程度”或“合理程度”。

一个精辟的总结（来自统计学家罗纳德·费舍尔）：
|   |解释|
|---|---|
|概率|是固定参数、变化数据时的密度|
|似然 |是固定数据、变化参数时的函数|

##### 3.2 怎么计算？

假定 $P(\bm{x}|c)$ 具有确定的概率分布形式，且由参数 $\theta_c$ 唯一确定，利用训练集 $D$ 来估计 $\theta_c$.

$D$ 中<span style = "background-color: #00bbffff"> 第 $c$ 类样本 </span>(好瓜or坏瓜) 集合 $D_c$ 的似然估计为： $P(D_c|\theta_c) = \prod\limits_{x\in D_c} P(\bm{x}|\theta_c)$.
取对数 $\hat{\theta_c} = \argmax\limits_{\theta_c}LL(\theta_c)=\log P(D_c|\theta_c) = \sum\limits_{\bm{x}\in D_c}\log P(\bm{x}|\theta_c)$.
假设 $d$ 个属性相互独立 $P(c|\bm{x})=\frac{P(c)P(\bm{x}|c)}{P(\bm{x})} = \frac{P(c)}{P(\bm{x})}\prod\limits_{i=1}^{d}P(x_i|c)$.
$P(x)$  对所有类别相同，于是对于$\bm{x}$ 的预测：
$h_nb(\bm{x})=\argmax \limits_{c\in \mathcal{Y}}P(c)\prod\limits_{i=1}^{d}P(x_i|c)$
其中 $P(c) = \frac{|D_c|}{|D|}$, $P(x_i|c) = \frac{|D_{c,x_i}|}{|D_c|}$
可以看到其实就是计算预测样本各个属性值在分类样本的占比乘积，然后选取最大的分类

##### 3.3 拉普拉斯修正

如果某个属性值在训练中没有出现过会出现0消除其他信息

令 $N$ 表示训练集 $D$ 中可能的类别数，$N_i$ 表示第 $i$ 个属性可能的取值数
$\hat{P(c)} = \frac{|D_c| +1}{|D|+N}$ , $\hat{P}(x_i|c) = \frac{|D_{c,x_i}|+1}{|D_c|+N_i}$

#### 4.半朴素贝叶斯

属性不一定满足"独立性假设"

- [x] 独依赖估计

假设每个属性在类别之外最多仅依赖一个其他属性
$P(c|\bm{x})\propto P(c)\prod\limits_{i=0}^{d}P(x_i|c,pa_i)$

两种常见的方法：
1.SPODE
假设所有属性都依赖同一属性，称为"超父"然后通过交叉验证等模型选择方法来确定超父属性.

2.TAN
以属性间的条件互信息为边的权重构建完全图，再利用最大帯权生成树算法，仅保留强相关属性间的依赖性
![alt text](img/{5D106379-4C22-4589-B7FA-0480D1A8355A}.png)