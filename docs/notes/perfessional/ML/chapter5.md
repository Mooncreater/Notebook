## 第五章
<h2 style="color: #2931d9ff; font-weight: normal;"> 神经网络</h2>

#### 1.神经元模型

##### 1.1概念
$y = f( \sum\limits_{i=1}^{n}w_ix_i - \theta)$, $f$ 为激活函数， $\theta$ 为阈值.

![alt text](img/{AC904666-000C-44DA-8858-B1C3FAFD31C2}.png)
##### 1.2激活函数

![alt text](img/{124B192D-167E-4E35-91F1-C100E9EDF9B8}.png)

#### 2.感知机与多层网络

##### 2.1 感知机
$y = f( \sum\limits_{i=1}^{n}w_ix_i - \theta)$, 可以容易实现 "与","或","非".
| 逻辑  |  实现  |
|---|----|
|“与” |  $w_1 = w_2 = 1, \theta = 2$  |
|“或”|  $w_1 = w_2 = 1, \theta =0.5$  |
|“非”|  $w_1 = -0.6, w_2 = 0, \theta = -0.5$  |

![alt text](img/{649F45C2-DBDD-4448-AD87-C59146BBA21F}.png) 

##### 2.2 感知机学习
训练样例 $(\bm{x},y)$ ,当前感知机输出 $\hat{y}$.
$w_i$ 变化 $\Delta w_i = \eta(y - \hat{y})x_i$, 学习率 $\eta \in (0,1)$ .

##### 2.3多层网络
求解非线性问题

![alt text](img/{E3917F84-E16B-405A-84A5-A2B5842587FC}.png)

通过学习来调整各层之间的 $w$ 和 $\theta$.
#### 3.误差逆传播算法(BP)
##### 3.1 概念

![alt text](img/{3053BF71-E8FF-4A4E-B9B6-6EEF04727FE7}.png)
##### 3.2 符号

|记号|解释|
|----|----    |
|训练集 $D$ | $D = \{ (\bm{x}_i,y_i)\},\bm{x}_i \in R^d,y_i \in R^l,i = 1,2,\cdots,m$|
|$\theta_j$| 输出层第 $j$ 个神经元阈值|
|$\gamma_h$| 隐含层第 $h$ 个神经元阈值|
| $v_{ih}$ |输入层和隐层神经元之间的连接权重|
| $w_{hj}$ |隐层和输出层神经元之间的连接权重|
| $b_h$ |第 $h$ 隐层输出 $b_h=f(\alpha_h - \gamma_h)$|
|$\alpha_h$|第 $h$ 隐层输入 $\alpha_h=\sum\limits_{i=1}^{d}v_{ih}x_i$|
|$\hat{y}_j^k $|第 $j$ 输出层输出 $\hat{y}_j^k = f(\beta_j - \theta_j)$|
|$\beta_j$|第 $j$ 输出层输入 $\beta_j=\sum\limits_{i=q}^{d}w_{hj}b_h$|

##### 3.3 思路
样例 $(\bm{x}_k,\bm{y}_k)$,实际网络输出 $\hat{y}_k$.

误差函数 $E_k = \frac{1}{2}\sum\limits_{j=1}^{l}(\hat{y}_j^k - y_j^k)^2$

- [x] 优化参数 
- [x] 目标负梯度方向调整参数
- [x] 局部最小

给定学习率 $\eta$, 考虑 $w_{hj}$ 的优化 $\Delta w_{hj}$ <span style = "background-color : #d4ff008a" >"对 $E_k$ 求偏导"</span>

$\Delta w_{hj} = - \eta \frac{\partial E_k}{\partial w_{hj}} = -\eta \frac{\partial E_k}{\partial \hat{y}_j^k} \cdot \frac{\partial\hat{y}_j^k}{\partial \beta_j}\cdot \frac{\partial \beta_j}{\partial w_{hj}}$

[^1]:Sigmoid 函数 $f(x) = \frac{1}{1+e^{-x}}$ 的导数 $f'(x) = f(x)(1-f(x))$

##### 3.4 关于单隐层神经元小结论

$b_h = \frac{\partial\beta_j}{\partial w_{hj}}$ 

$g_j = \frac{\partial E_k}{\partial \hat{y}_j^k} \cdot \frac{\partial\hat{y}_j^k}{\partial \beta_j} = (\hat{y}_j^k -y_j^k)\cdot f'(\beta_j -\theta_j)$ [^1] $=(\hat{y}_j^k -y_j^k)\cdot \hat{y}_j^k(1-\hat{y}_j^k)$

$e_h = -\frac{\partial E_k}{\partial b_h}\cdot \frac{\partial b_h}{\partial \alpha_h}= -\sum\limits_{j=1}^{l}\frac{\partial E_k}{\partial \beta_j}\cdot \frac{\partial\beta_j}{\partial b_h}\cdot f'(\alpha_h-\gamma_h) = \sum\limits_{j=1}^{l}g_i w_{hj}b_h (1-b_h)$

$\Delta w_{hj} = \eta g_j b_h$

$\Delta \theta_j = -\eta g_j$

$\Delta v_{ih} = \eta e_h x_i$

$\Delta \gamma_h = -\eta e_h$

##### 3.5 累计 BP 算法
优化目标是整个训练集累计误差 $E = \frac{1}{m} \sum\limits_{k=1}^{m} E_k$.
读取<u>整个训练集</u>一遍再进行参数更新

##### 3.6 缓解过拟合
早停：若训练误差降低，但验证误差升高，则停止训练
正则化：$E = \lambda\frac{1}{m}\sum\limits_{k=1}^{m}E_k +(1-\lambda)\sum\limits_{i}w_i^2$

#### 4.其他神经网络

RBF 网络：单隐层前馈神经网络, 它使用径向基函数作为隐层神经元激活函数, 而输出层则是隐层神经元输出的线性组合.
ART 网络：输出神经元一个激活其他全被抑制.