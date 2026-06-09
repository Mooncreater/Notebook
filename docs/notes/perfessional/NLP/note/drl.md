---	
comments: true	
---	
	
# 深度强化学习 (Deep RL)	
	
强化学习：智能体通过与环境交互（试错）学习最优策略，以获得最大累积奖励。	
	
!!! tip "核心要点"	
    RL 三大范式：**Value-based**（学评价函数，如 DQN）、**Policy-based**（直接学策略，如 Policy Gradient）、**Actor-Critic**（两者结合，如 A3C）。DQN 用神经网络解决维度灾难，是深度 RL 的里程碑。	
	
## 1. 强化学习基础	
	
### 起源	
	
- **心理学**：行为主义（Law of Effect, Thorndike 1911）—— 试错学习	
- **控制论**：Bellman 方程 (1957)、MDP、Policy Iteration	
- **集大成者**：Richard S. Sutton (Alberta 大学)	
	
### 基本概念	
	
在 $t$ 时刻：	
	
- **Agent** 执行 action $a_t$	
- **Environment** 返回 observation $o_{t+1}$ 和 reward $r_{t+1}$	
- 目标：最大化累积奖励 $\sum_t \gamma^t r_t$	
	
### RL 分类	
	
| 类型 | 做法 | 代表算法 |	
|:---|:---|:---|	
| **Value-based** | 估计期望回报，选价值最大的动作 | Q-Learning, DQN |	
| **Policy-based** | 直接优化策略函数 | Policy Gradient |	
| **Actor-Critic** | 策略 + 价值相结合 | A3C |	
| **Model-based** | 先学环境模型再规划 | Dyna-Q |	
	
## 2. Value-based RL：Q-Learning 与 DQN	
	
### Q-Learning	
	
学习 Q 函数 $Q(s, a)$：在状态 $s$ 执行动作 $a$ 的期望累积奖励。	
	
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right]$$	
	
### 维度灾难	
	
传统 Q-Learning 用表格存储 $Q(s, a)$，状态空间大时无法处理。	
	
### DQN (Deep Q-Network)	
	
**核心创新**：用深度神经网络近似 Q 函数。	
	
2013 → 2015：DeepMind 的 DQN 在 48 个 Atari 游戏中超越人类水平，登上 Nature 封面。	
	
#### 关键技术	
	
1. **Experience Replay**：存储经验四元组 $(s, a, r, s')$，随机采样训练，打破数据相关性	
2. **Target Network**：独立的目标 Q 网络，周期性同步，稳定训练	
	
#### DQN 改进	
	
| 改进 | 方法 |	
|:---|:---|	
| **Double DQN** | 解耦动作选择与评估，缓解 Q 值高估 |	
| **Prioritized Replay** | 根据 TD-error 加权采样重要经验 |	
| **Dueling Network** | 分离状态价值 $V(s)$ 和优势函数 $A(s,a)$ (ICML 2016 Best Paper) |	
	
## 3. Policy-based RL：Policy Gradient	
	
### 思想	
	
直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望回报。好的动作（高回报）提高概率，坏的动作降低概率。	
	
### Robot in a Room 案例	
	
| 环境 | 策略 |	
|:---|:---|	
| 确定性动作 | 最短路径 |	
| 随机动作 + 惩罚每步 | 绕开危险区域 |	
| 奖励每步 | 最长路径（拖延） |	
	
结论：环境模型和奖励结构对最优策略影响巨大。	
	
## 4. Actor-Critic 方法	
	
### 为什么需要 Critic？	
	
Policy Gradient 的回报方差大。引入 Critic 评估"当前状态/动作有多好"。	
	
### A3C (Asynchronous Advantage Actor-Critic)	
	
- **Actor**：输出动作概率分布（策略网络）	
- **Critic**：输出状态价值 $V(s)$（价值网络）	
- **Advantage**：$A(s, a) = Q(s, a) - V(s)$，用优势函数替代原始奖励，减少方差	
	
### 异步训练	
	
多个 worker 并行在各自环境副本中收集经验，异步更新全局网络参数。	
	
### TD vs MC	
	
| 方法 | 更新时机 | 特点 |	
|:---|:---|:---|	
| **MC (蒙特卡洛)** | Episode 结束后 | 无偏但方差大 |	
| **TD (时序差分)** | 每步 | 有偏但方差小，更快 |	
	
!!! warning "常见误区"	
    RL 中 RLHF (Reinforcement Learning from Human Feedback) 是大模型对齐的关键技术。但这里的强化学习基础主要用于理解后续的对齐方法（PPO、DPO 等）。	
	
!!! danger "考试重点"	
    Q-Learning 的更新公式、DQN 的两个关键技术（Experience Replay + Target Network）、Policy Gradient 的核心思想、Actor-Critic 架构。	