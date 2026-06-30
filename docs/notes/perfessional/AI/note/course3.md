---
comments: true
---

# 托马斯算法（Tomasulo Algorithm）

目标：解决**乱序执行**中的数据依赖问题。类似 Dataflow 模型 —— "fire" 指令时检查所有输入是否准备好，准备好就执行，否则等待。

!!! tip "核心要点"
    Tomasulo 通过三件法宝实现乱序执行：**寄存器重命名**消除假依赖、**保留站**等待操作数、**CDB 广播**传递结果。

## 为什么需要 Tomasulo？

仅有 ROB（Reorder Buffer）的流水线是 **In-order Dispatch**：即使 ROB 消除了假依赖（WAR/WAW），但当真依赖（RAW）导致某条指令等待时，**后续所有独立指令也被阻塞**。

```
IMUL  R3 ← R1, R2     ; 4 cycles, R3 是后续指令的依赖
ADD   R3 ← R3, R1      ; 等待 R3，阻塞！
ADD   R1 ← R6, R7      ; 独立指令，也被阻塞！
IMUL  R5 ← R6, R8      ; 独立指令，也被阻塞！
ADD   R7 ← R3, R5      ; 等待 R3
```

Tomasulo 的核心思想：**把依赖指令"移开"，让独立指令先执行**。

> Key Idea: Move the dependent instructions out of the way of independent ones.

:arrow_right: In-order dispatch → 16 cycles；Out-of-order dispatch → 12 cycles

---

## 三大组件详解

### 1. Register Rename Table（寄存器重命名表 / RAT）

**数据结构含义**：

寄存器重命名表是一个**映射表**，将架构寄存器名映射到 **Tag**（标签），用 Tag 代替寄存器名来消除假依赖。

| 字段 | 含义 |
|------|------|
| `valid` | = 1：寄存器的值已准备好，可直接从寄存器文件读取 |
| | = 0：寄存器值还在计算中，需等待对应 Tag 的广播 |
| `tag` | 当 valid=0 时，指向**正在产生该寄存器值的那条指令所在保留站/Tag** |

**何时更新**：

- **Issue 阶段（写 RAT）**：指令发射时，如果该指令有目的寄存器 Rd，则在 RAT 中将 Rd 的 `valid` 置 0，`tag` 设为该指令的保留站编号。这表示"Rd 的新值正在由 Tag X 生产"。
- **WB 阶段（更新 RAT）**：指令完成并通过 CDB 广播结果时，如果 RAT 中该寄存器的 tag 与广播的 tag 匹配，则将 `valid` 置 1，清除 tag。

**如何更新（详细流程）**：

```
Issue 时：
  if (指令有目的寄存器 Rd):
    RAT[Rd].valid = 0
    RAT[Rd].tag   = 该指令的保留站ID

  读源操作数 Rs1/Rs2：
    if (RAT[Rs].valid == 1):
      直接读寄存器文件的值 → 放入保留站
    else:
      记录 RAT[Rs].tag → 放入保留站（等待 CDB 广播匹配）

WB 时：
  当某指令在 CDB 上广播 {tag, value}：
    for each 寄存器 R in RAT:
      if (RAT[R].tag == 广播的tag):
        RAT[R].valid = 1
        RAT[R].tag   = 清除
```

!!! danger "考试重点"
    RAT 的核心作用：**消除 WAR 和 WAW 假依赖**。假依赖的本质是"多个指令用了同一个寄存器名，但这些值之间没有真正的数据流关系"。RAT 给每个生产者分配不同 Tag，让它们"改名换姓"，从而不再冲突。

**RAT 与 ROB 重命名的区别**：

- ROB 重命名：用 ROB Entry ID 替代寄存器名，提交时恢复
- RAT（Tomasulo中）：跟踪"当前最新生产该寄存器的指令 Tag"
- 现代处理器中二者配合：ROB Entry ID 作为 Tag 使用

---

### 2. Reservation Station（保留站）

**数据结构含义**：

保留站是**每条指令等待操作数的"休息区"**。每个功能单元（如 ALU、乘法器、Load/Store 单元）关联一组保留站。

每条保留站条目包含：

| 字段 | 含义 |
|------|------|
| `Op` | 操作码（ADD / MUL / LD / ST ...） |
| `Vj` | 源操作数1：若已就绪则为**值**，否则为等待的 **Tag** |
| `Qj` | 源操作数1的 Tag（与 Vj 互斥：Vj 有值时 Qj=0） |
| `Vk` | 源操作数2：若已就绪则为**值**，否则为等待的 **Tag** |
| `Qk` | 源操作数2的 Tag（与 Vk 互斥） |
| `Busy` | 该保留站是否被占用 |
| `Dest Tag` | 该指令产生结果的 Tag（用于 WB 阶段匹配） |
| `A` | （仅 Load/Store）存储地址信息 |

**何时更新**：

- **Issue 阶段**：分配保留站，填入 Op、Vj/Qj、Vk/Qk（从 RAT 查询获得）、设置 Busy=1、设置 Dest Tag
- **CDB 广播匹配**：每当 CDB 广播 {tag, value}，所有保留站检查自己的 Qj/Qk。若 Qj == 广播的 tag，则 Vj = value, Qj = 0（操作数就绪！）。Qk 同理。
- **Dispatch（发射执行）**：当 Vj 和 Vk 都有值（即 Qj=Qk=0）时，指令"fire"——进入执行单元。
- **WB 阶段**：执行完成，释放保留站（Busy=0）

**如何更新（详细流程）**：

```
Issue 时分配保留站：
  RS[分配].Op   = 操作码
  RS[分配].Busy = 1
  RS[分配].Dest = 该指令Tag
  
  读 RAT 获取源操作数：
    if (RAT[Rs1].valid): RS.Vj=寄存器值, RS.Qj=0
    else:                RS.Qj=RAT[Rs1].tag, RS.Vj=无效
    
    if (RAT[Rs2].valid): RS.Vk=寄存器值, RS.Qk=0
    else:                RS.Qk=RAT[Rs2].tag, RS.Vk=无效

CDB 广播时更新保留站：
  当 CDB 上出现 {tag, value}：
    for each 保留站条目 RS[i]:
      if (RS[i].Qj == tag): RS[i].Vj = value; RS[i].Qj = 0
      if (RS[i].Qk == tag): RS[i].Vk = value; RS[i].Qk = 0

发射条件检查：
  if (RS[i].Qj == 0 && RS[i].Qk == 0): 
    该指令可以执行！→ 送入功能单元
```

!!! danger "考试重点"
    保留站的核心：**分布式调度**。没有中央控制器，每个保留站自己监听 CDB、自己判断操作数是否就绪。Qj/Qk=0 就是发射信号。这是 Tomasulo 和 Scoreboard 的本质区别。

---

### 3. Common Data Bus（公共数据总线 / CDB）

**含义**：一条共享广播总线。每次有指令完成时，将结果 `{tag, value}` 广播到 CDB。**所有保留站和 RAT 同时监听 CDB**。

!!! tip "精妙设计"
    CDB 广播意味着结果**直接从完成单元传递到等待单元**，不需要经过寄存器文件。这是 Tomasulo 高效的关键——避免了"写完寄存器→再读寄存器"的延迟。

CDB 广播后：
1. 所有保留站：匹配 Qj/Qk → 更新 Vj/Vk
2. RAT：匹配 tag → 更新 valid=1
3. 如果该指令在 ROB 中有条目 → 标记完成

---

## 算法三阶段（完整流程）

### Stage 1: Issue（发射 / ID 阶段）—— 按程序顺序

1. 从指令队列取**下一条指令**（按程序顺序）
2. 检查是否有空闲保留站：**若无 → 停顿（结构冲突）**
3. 查 RAT 获取源操作数：
   - `valid=1` → 直接读寄存器值放入保留站
   - `valid=0` → 将 tag 放入保留站的 Qj/Qk
4. 如果指令有目的寄存器 Rd：**更新 RAT** → RAT[Rd].valid=0, RAT[Rd].tag=该指令Tag
5. 在 ROB 中分配条目（按程序顺序）

### Stage 2: Execute（执行 / EX 阶段）—— 乱序执行

- 保留站持续监听 CDB，当 Qj=0 且 Qk=0 时 → **fire**
- 多条指令可在不同功能单元中**同时执行**
- 多条就绪指令可**乱序**进入执行
- Load/Store 需计算地址后才能访问内存

### Stage 3: Write Back（写回 / WB 阶段）

1. 功能单元完成计算，获得结果 `value`
2. 将 `{该指令Tag, value}` 广播到 CDB
3. 释放保留站
4. ROB 标记该条目完成

---

## 完整示例：逐拍跟踪

假设：ADD=1cycle, MUL=4cycles, 保留站充足

```
指令序列：
I1: MUL  R3 ← R1, R2
I2: ADD  R3 ← R3, R1    ; RAW 依赖 I1
I3: ADD  R1 ← R6, R7    ; WAR 依赖 I2（假依赖！）
I4: MUL  R5 ← R6, R8
I5: ADD  R7 ← R3, R5    ; RAW 依赖 I1 和 I4
```

| Cycle | Issue | Execute | CDB/WB | 说明 |
|-------|-------|---------|--------|------|
| 1 | I1 | - | - | RAT[R3].tag=I1, 保留站I1: Vj=R1, Vk=R2 |
| 2 | I2 | I1(MUL) | - | RAT[R3].tag=I2(假依赖被RAT消除!), I2: Qj=I1_tag, Vk=R1 |
| 3 | I3, I4 | I1(MUL) | - | I3 独立，I4 独立，不阻塞！(RAT[R1].tag=I3) |
| 4 | I5 | I1(MUL) | - | I5: Qj=I2_tag, Qk=I4_tag |
| 5 | - | I1(MUL完) | I1:{tag,val}→CDB | I2的Qj匹配→Vj=val,I2就绪！ |
| 6 | - | I2(ADD), I3(ADD) | I2:{tag,R3_new}→CDB | I2完成；I5的Qj匹配→就绪一个操作数 |
| 7 | - | I4(MUL) | I3完成 | I3不依赖任何人 |
| 8-10 | - | I4(MUL) | - | MUL 4周期到 cycle 10 |
| 10 | - | I4(MUL完) | I4:{tag,val}→CDB | I5的Qk匹配→I5就绪 |
| 11 | - | I5(ADD) | I5完成 | - |

:arrow_right: **关键观察**：I3 在 Cycle 2 就已发射，不等待 I1/I2。这就是乱序执行的力量。

---

## Tomasulo 中的 Load/Store 处理

- **Load**：需要计算地址后才能访存。地址就绪 + 无更早的未完成 Store → 可以执行
- **Store**：也必须计算地址。写内存只在 Commit 阶段进行（精确异常要求）
- Load/Store 之间通过地址比较检测 RAW/WAR/WAW 内存依赖

---

## 与 Scoreboard 对比

| | Scoreboard | Tomasulo |
|--|-----------|----------|
| 假依赖处理 | **停顿等待** | **寄存器重命名消除** |
| 结果传递 | 通过寄存器文件 | **CDB 广播**（直连） |
| 调度方式 | 集中式控制 | **分布式保留站** |
| WAR 处理 | 必须等读完成 | 重命名后无需等待 |
| WAW 处理 | 必须等前一条写完 | 重命名后无需等待 |
| 复杂度 | 较简单 | 较复杂但性能更高 |

---

## 为什么 Tomasulo 有效？

- **寄存器重命名（RAT）**：消除 WAR / WAW 假依赖 → 指令不会因"同名寄存器"而阻塞
- **保留站 + CDB 广播**：分布式调度，结果直接从完成单元传到等待单元 → 不需经过寄存器文件
- **乱序执行**：独立指令可超越被阻塞的依赖指令先执行 → 容忍长延迟操作

---

## 现代 OoO 处理器的"双峰"结构

```
       ┌──────────────┐
       │  Reservation │  ← Hump 1: 调度窗口（乱序发射）
       │   Stations   │
       └──────┬───────┘
              │
    ┌─────────┴─────────┐
    │  Functional Units │  ← 乱序执行
    └─────────┬─────────┘
              │
       ┌──────┴───────┐
       │   Reorder     │  ← Hump 2: 重排序（顺序提交）
       │    Buffer     │
       └──────────────┘
```

- **Hump 1（保留站）**：乱序发射，容忍延迟
- **Hump 2（ROB）**：顺序提交，保证精确异常

---

## 精确异常实现

- ROB 确保指令按程序顺序提交
- 发生异常时：ROB 中该指令之前的所有指令已完成提交，之后的全被 Flush
- 恢复方法：拷贝 Architectural Register File → Frontend Register File（现代做法）

---

## 限制与思考

- **指令窗口大小**决定能容忍多少周期的延迟
- 若某指令（如 Cache Miss）延迟 1000 cycles，窗口不够大则仍会阻塞
- 窗口大小受限于：保留站数量、ROB 深度、寄存器数量

!!! warning "常见误区"
    Tomasulo 本身不处理精确异常！精确异常由 ROB 保证。Tomasulo + ROB 才是完整的乱序执行方案。
