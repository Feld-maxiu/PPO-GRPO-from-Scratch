# PPO & GRPO from Scratch

从零实现 PPO (Proximal Policy Optimization) 和 GRPO (Group Relative Policy Optimization) 强化学习算法，用于大语言模型的训练。

## 目录

- [简介](#简介)
- [PPO 算法](#ppo-算法)
- [GRPO 算法](#grpo-算法)
- [PPO vs GRPO 对比](#ppo-vs-grpo-对比)
- [数学原理](#数学原理)

## 简介

本项目提供了 PPO 和 GRPO 两种强化学习算法的清晰实现，帮助理解：

- **PPO**: OpenAI 提出的经典策略梯度算法，通过裁剪目标函数限制策略更新幅度
- **GRPO**: DeepSeek 提出的改进算法，使用组内相对奖励代替价值网络，更适合大模型训练

两个算法都应用于 GSM8K 数学推理任务，训练语言模型生成带 `\boxed{}` 格式的答案。

## PPO 算法

### 核心思想

PPO 通过以下步骤优化策略：

1. **收集经验**: 使用当前策略生成响应，计算奖励
2. **计算优势**: 使用 GAE (Generalized Advantage Estimation) 估计优势函数
3. **裁剪更新**: 限制策略更新幅度，防止策略崩溃
4. **价值学习**: 同时训练价值网络估计期望回报

### 训练流程

```
for iteration in range(num_iterations):
    # 1. Rollout: 收集经验
    memories = collect_rollouts(agent, dataset)
    
    # 2. 计算 GAE 优势
    advantages = compute_gae(rewards, values)
    
    # 3. 更新策略和价值网络
    for epoch in range(ppo_epochs):
        loss = ppo_loss(old_log_probs, new_log_probs, advantages)
        loss.backward()
        optimizer.step()
```


## GRPO 算法

### 核心思想

GRPO 是 PPO 的改进版本，主要创新：

1. **无需价值网络**: 使用组内相对奖励作为基线
2. **组采样**: 每个问题采样 G 个回答，形成一组进行比较
3. **相对优势**: 通过组内标准化计算优势
4. **KL 惩罚**: 约束策略不偏离参考模型太远

### 训练流程

```
for iteration in range(num_iterations):
    for each question:
        # 1. Group Rollout: 采样 G 个回答
        group_memories, rewards = collect_group_rollouts(agent, query, group_size=G)
        
        # 2. 计算相对优势
        advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
        
        # 3. 更新策略（带 KL 惩罚）
        for epoch in range(ppo_epochs):
            loss = grpo_loss(old_log_probs, new_log_probs, ref_log_probs, advantages)
            loss.backward()
            optimizer.step()
```


## PPO vs GRPO 对比

| 特性 | PPO | GRPO |
|------|-----|------|
| **价值网络** | 需要训练 | 不需要 |
| **基线估计** | 价值函数 V(s) | 组内奖励均值 |
| **优势计算** | GAE | (r_i - μ) / σ |
| **参考模型** | 无 | 有（用于 KL 惩罚）|
| **采样方式** | 单样本 | 组采样 (G个) |
| **显存占用** | 较高 | 较低 |
| **适用场景** | 通用 RL | 大模型训练 |

### 优势对比

**PPO 优势:**
- 经典成熟，广泛应用
- 价值网络提供稳定的基线估计

**GRPO 优势:**
- 无需训练价值网络，节省显存
- 组内比较天然适合大模型的多样化采样
- KL 惩罚防止模型偏离太远

## 代码结构

```
.
├── ppo_from_scratch.py    # PPO 算法实现
├── grpo_from_scratch.py   # GRPO 算法实现
├── gsm8k/                 # GSM8K 数据集
└── README.md             # 本文件
```



### 环境要求

```bash
pip install torch transformers datasets
```

## 数学原理

### PPO 目标函数

PPO-Clip 目标函数：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比
- $\hat{A}_t$ 是优势函数估计（GAE）
- $\epsilon$ 是裁剪超参数（通常 0.1 或 0.2）

### GAE (Generalized Advantage Estimation)

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
$$

其中 TD 误差：

$$
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

### GRPO 相对优势

对于组内 G 个样本：

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, r_2, ..., r_G\})}{\text{std}(\{r_1, r_2, ..., r_G\}) + \epsilon}
$$

### GRPO 损失函数

$$
L_{GRPO}(\theta) = \mathbb{E} \left[ \min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t) - \beta D_{KL}(\pi_{ref} || \pi_\theta) \right]
$$

其中 KL 散度近似：

$$
D_{KL}(\pi_{ref} || \pi_\theta) \approx \log \pi_{ref}(a|s) - \log \pi_\theta(a|s)
$$

## 实现细节

### 奖励函数

两个算法使用相同的奖励函数，检查 `\boxed{}` 内的答案：

```python
def compute_rewards(completions, ground_truth):
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]
```

### 数值稳定性

- 使用 `log_softmax` 代替 `softmax` + `log`
- 概率裁剪到 `[1e-10, 1.0]` 范围
- 梯度裁剪 (`clip_grad_norm_=0.5`)

### 混合精度训练

支持 FP16 混合精度训练，节省显存：

```python
from torch.amp import autocast, GradScaler

with autocast(device_type='cuda', enabled=True):
    outputs = model(input_ids)
    
scaler = GradScaler()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 参考文献

1. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. **GRPO**: DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)
3. **GAE**: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2016)

## 许可证

MIT License
