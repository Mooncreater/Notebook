---
comments: true
--- 

# 超标量与向量化


## 超标量
IPC(Instructions Per Cycle) 每时钟周期执行的指令数  
处理器能够在同一个时钟周期内，同时启动（或发射）多条独立的指令。 
### 优劣

优点： 
- 高指令吞吐    
- 高IPC     

缺点：
- 高复杂依赖检查    
- 更多的硬件依赖    

### Can 超标量 影响 Roofline 模型的性能吗？

#### Roofline 模型

**算力** ：计算量 / 访存量
- 单位： FLOPS/Byte 每字节数据做多少次浮点计算
- 性能： 每秒完成的计算量


## 向量化

定义：同时对多个数据进行相同的操作，使用单条指令处理多个数据元素。

### 分类

SISD（单指令单数据）: 传统的处理器架构，每条指令处理一个数据元素。  
SIMD（单指令多数据）: 向量化处理器架构，每条指令处理多个数据元素。  
MISD（多指令单数据）: 多条指令处理一个数据元素。    
MIMD（多指令多数据）: 多条指令处理多个数据元素。    


![alt text](../img/{D6AACECD-5C6D-445A-AD80-9849E78DE6C1}.png)

![alt text](../img/image5.png)

### SIMD 

对于计算A[6..0] + B[6..0]

标量    
![alt text](../img/{6546BD8C-C5F6-4E7E-9939-DD4F6A62D9FE}.png)

SIMD    
![alt text](../img/{12F35AED-7097-466B-B55A-FF0B5834B6EA}.png)



## 多线程



## 多核

摩尔定律

put multiple cores on the same die


