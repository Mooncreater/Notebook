## 第六章
<h2 style="color: #2931d9ff; font-weight: normal;"> 支持向量机</h2>

#### 1.间隔与支持向量
什么是支持向量？

- [x] 支撑整个"道路"宽度的点
- [x] 边缘点
- [x] 满足 $y_i(\bm{w}^T\bm{x}_i +b) = 1$ 的点

样本 $(\bm{x}_i,y_i)$, 其中 $y_i = \pm 1$.

我们需要找到划分函数 $\hat{y} = \bm{w}^T\bm{x} + b$ 对样本进行分类

在离散的样本中，$\hat{y} > \theta_1$ 为正例， $\hat{y} < \theta_2$ 为正例,中间不希望有样本

临界线 $\bm{w}^T\bm{x} + b = \theta_1$, $\bm{w}^T\bm{x} + b = \theta_2$.

齐次归一临界线：$\bm{w'}^T\bm{x} + b' = 1$, $\bm{w'}^T\bm{x} + b' = -1$.

即在新的划分函数 $\hat{y}>1$ 正例，$\hat{y} < -1$ 反例.

寻找参数 $\bm{w}$ 和 $b$,使得间隔 $\gamma = \frac{2}{\|\bm{w}\|}$ 最大，即求 $\frac{\|\bm{w}\|}{2}$最小.

满足所有点划分正确 $y_i(\bm{w}^T\bm{x}_i +b) \geq 1, i=1,2,\cdots,m$.

![alt text](img/{FD915AD6-8DC9-42BE-8ECE-A456B22D28DA}.png)

#### 2.对偶问题

现在的问题：求 $\frac{1}{2}\|\bm{w}\|$ 最小值 $\iff$ 求 $\frac{1}{2}\|\bm{w}\|^2$ 最小值.
约束条件：$y_i(\bm{w}^T\bm{x}_i +b) \geq 1$ $\iff$ $-(y_i(\bm{w}^T\bm{x}_i +b) - 1)\leq 0$
##### 拉格朗日乘子法
拉格朗日函数： $L(\bm{w},b,\bm{\alpha}) = \frac{1}{2}\|\bm{w}\|^2 - \sum\limits_{i=1}^{m}\alpha_i\big(y_i(\bm{w}^T\bm{x}_i+b)-1\big)$

分别对 $\bm{w}$ , $b$ 求偏导取零有：$\bm{w} = \sum\limits_{i=1}^{m}\alpha_iy_i\bm{x}_i$,$\sum\limits_{i=1}^{m}\alpha_iy_i = 0$.

问题转化为：$\min\limits_{\alpha}\frac{1}{2}\sum\limits_{i=1}^{m}\sum\limits_{j=1}^{m}\alpha_i\alpha_jy_iy_j\bm{x}_i^T\bm{x}_j -\sum\limits_{i=1}^{m}\alpha_i$

最终模型：$f(x) = \bm{w}^T\bm{x} + b = \sum\limits_{i=1}^{m}\alpha_i y_i \bm{x}_i^T\bm{x} + b$

$KKT$ 条件：

$$\begin{cases}
\alpha_i \geq 0\\
y_if(\bm{x}_i) \geq 1\\
\alpha_i(y_if(\bm{x}_i) - 1) = 0
\end{cases}$$

解的稀疏性：$y_if(\bm{x}_i) > 1$ 则 $\alpha_i = 0$ <span style = "background-color : #00bbffff">最终模型仅与支持向量有关</span>

##### SMO
现在的约束：$\sum\limits_{i=1}^{m}\alpha_iy_i = 0$

###### 思路：
1.选取需要更新的变量 $\alpha_i$ 和 $\alpha_j$
2.固定其他 $\alpha$ 参数

#### 3.核函数
无法找到能划分两类样本的超平面？
- [x] 映射高维空间
![alt text](img/{86F840BF-24E4-4DD1-892F-A33B9CA9CDE1}.png)

观察最终模型：$f(x) = \bm{w}^T\bm{x} + b = \sum\limits_{i=1}^{m}\alpha_i y_i \bm{x}_i^T\bm{x} + b$,仅与内积有关.
所以将 $\bm{x}$ 映射为 $\phi(\bm{x})$ 也只和 $\phi(\bm{x_i})^T\phi(\bm{x_j})$ 有关

##### 3.1 基本想法
不显式设计核映射，而是设计核函数：$\kappa(\bm{x}_i,\bm{x}_j) = \phi(\bm{x}_i)^T\phi(\bm{x}_j)$.

##### 3.2 常见核函数

| 名称 | 表达式 | 参数 |
| :--- | :--- | :--- |
| **线性核** | \(\kappa(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i^T \mathbf{x}_j\) |  |
| **多项式核** | \(\kappa(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i^T \mathbf{x}_j)^d\) | \(d \geq 1\) 为多项式的次数 |
| **高斯核** | \(\kappa(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2\sigma^2}\right)\) | \(\sigma > 0\) 为高斯核的带宽 (width) |
| **拉普拉斯核** | \(\kappa(\mathbf{x}_i, \mathbf{x}_j) = \exp\left(-\frac{\|\mathbf{x}_i - \mathbf{x}_j\|}{\sigma}\right)\) | \(\sigma > 0\) |
| **Sigmoid核** | \(\kappa(\mathbf{x}_i, \mathbf{x}_j) = \tanh(\beta \mathbf{x}_i^T \mathbf{x}_j + \theta)\) | \(\tanh\) 为双曲正切函数，\(\beta > 0\)，\(\theta < 0\) |

#### 4.软间隔与正则化
还是难以找到线性可分？
- [x] 软间隔
- [x] 允许支持向量机有一些样本不满足约束
![alt text](img/{D4978882-0F8A-45EE-80B2-DD1B426CBA75}.png)

##### 4.1 0/1损失函数

最大化间隔同时，不满足约束的样本尽可能少 
$\min\limits_{\bm{w},b}\frac{1}{2}\|\bm{w}\|^2 + C\sum\limits_{i=1}^{m}l_{0/1}\big(y_i(\bm{w}^T\phi(\bm{x}_i)+b)-1\big)$

$l_{0/1} = \begin{cases}
1 & z < 0\\
0  &othrewise
\end{cases}$

##### 4.2 正则化
一般形式：
$\min\limits_{f} \Omega(f) + C\sum\limits_{i=1}^{m}l(f(\bm{x}_i),y_i)$
前者：结构风险，描述模型的一些性质
后者：经验风险，描述模型和训练数据的契合度

#### 5.支持向量回归
允许模型输出和实际输出存在 $2\epsilon$ 偏差
![alt text](img/{E78D9D82-69BD-44C0-99CF-47CDBCFA73B7}.png)
落入中间 $2\epsilon$ 样本不计入损失
