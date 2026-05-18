---
comments: true
---


# 托马斯算法

目标：解决乱序执行,就像 Dataflow ，"fire"指令时，检查所有输入是否准备好，如果准备好就执行，否则等待。

## Reservation Station

**Key Idea**: Move the dependent instructions out of the way of indenpendent ones.


## 三个组成

- Register Rename Table
- Reservation Station
- Common Data Bus 公共数据总线

**Register Rename Table**   
valid = 1: 代表这个寄存器的值已经准备好了，可以直接使用      
valid = 0: 代表这个寄存器的需要Tag计算，还在计算中  

![alt text](../img/{588FCCED-6A66-4325-98BD-15802ACC7202}.png) 

**Reservation Station**     
![alt text](../img/image2.png)

**Common Data Bus**     
进行广播，每次有指令完成时，都会将结果广播到公共数据总线，所有等待这个结果的指令都可以从总线上获取数据并继续执行。

## 算法
1. ID阶段：
2. RS阶段：
3. EX阶段：

