---
comments : true
---

# 深度学习
- [x] 定义 a set of function
- [x] 评价函数优良性
- [x] 选择最佳函数

## 机器学习

机器学习  $\approx$  寻找一个函数 
数据集训练模型   

## 神经网络

$z= a_1w_1 + ... +a_kw_k +... +a_Kw_K + b$.    
***Sigmoid Function***: $\sigma(z) = \frac{1}{1+e^{-z}}$.  
向前传递 $\sigma(z) \to a$.  

### 全连接神经网络

![alt text](../img/{BD27EC42-A720-4C26-B49C-CA6E271D0E61}.png)

### 输出层
作为神经网络最后一层的激活函数
$Softmax(z_i) = \frac{e^{z_i}}{\sum\limits_{j=1}^{K}e^{z_j}}$

### 损失
