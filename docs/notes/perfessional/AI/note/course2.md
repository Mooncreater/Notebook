---	
comments: true	
---	
	
# 流水线与乱序执行	
	
!!! tip "核心要点"	
    流水线通过并行提高吞吐，但引入三种冲突。数据冲突用转发解决，控制冲突用分支预测，乱序执行 + ROB 保证顺序提交。	
	
## 流水线基础	
	
经典五级流水线：	
	
| 阶段 | 名称 | 功能 |	
|------|------|------|	
| IF | Instruction Fetch | 取指令 |	
| ID | Instruction Decode | 译码 + 读寄存器 |	
| EX | Execute | 执行运算 |	
| MEM | Memory Access | 访存 |	
| WB | Write Back | 写回寄存器 |	
	
## 流水线冲突	
	
### 结构冲突（Structural Hazard）	
	
硬件资源不够同时支持多条指令。	
	
:arrow_right: 增加硬件（如分离指令/数据 Cache）	
	
### 数据冲突（Data Hazard）	
	
指令之间存在数据依赖：	
	
- **RAW（Read After Write）**：真正依赖，必须等待	
- **WAR（Write After Read）**：反依赖（假依赖）	
- **WAW（Write After Write）**：输出依赖（假依赖）	
	
:arrow_right: 解决：**转发（Forwarding / Bypassing）**—— 不等 WB 阶段，EX 阶段结果直接转发到下一条指令。	
	
### 控制冲突（Control Hazard）	
	
分支指令导致 PC 不确定取哪条指令。	
	
:arrow_right: 解决：**分支预测**（Branch Prediction）—— 静态预测（如 predict-not-taken）或动态预测（如 2-bit 饱和计数器）。	
	
!!! warning "常见误区"	
    RAW 是真依赖，必须等。WAR 和 WAW 是假依赖，可以通过**寄存器重命名**彻底消除，不需要等待。	
	
## Reorder Buffer（重排序缓冲）	
	
定义：指令完成后，结果暂存于 ROB 中。ROB 确保指令**按程序顺序提交**，即使执行顺序不同。	
	
- [x] 执行时允许乱序完成	
- [x] 提交时严格按程序顺序	
	
从而达成：**顺序发射 → 乱序执行 → 顺序提交**	
	
![alt text](../img/image.png)	
	
### ROB Entry	
	
包含：指令类型、目标寄存器、结果值、完成状态。	
	
![alt text](../img/{971D100F-57EC-4D9E-82AF-FEE6DF32E46E}.png)	
	
### 寄存器重命名	
	
用 ROB 条目 Tag 替代寄存器名，消除假依赖（WAR / WAW）。