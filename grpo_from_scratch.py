"""
GRPO (Group Relative Policy Optimization) 从头实现
基于 DeepSeek-R1 的 GRPO 算法

GRPO 核心思想：
1. 组采样：对每个问题采样 G 个回答
2. 相对优势：使用组内奖励的均值和标准差来计算优势，无需价值网络
3. 策略更新：使用 PPO-clip 目标函数更新策略
4. KL 惩罚：约束新策略与参考策略的 KL 散度，防止策略偏离太远

与 PPO 的主要区别：
- PPO 使用价值网络估计基线，GRPO 使用组内相对奖励作为基线
- GRPO 不需要训练价值网络，节省显存和计算
- GRPO 引入参考模型和 KL 散度惩罚
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import re
from tqdm import tqdm
import numpy as np


USE_GPU = True
USE_AMP = True
SYSTEM_PROMPT = (
    "Please solve the problem and answer in the format of 'The answer is...', no process required. "
    "Always include the final numeric answer inside \\boxed{}."
)


class GRPOAgent(nn.Module):
    """
    GRPO 智能体：包含策略网络和参考模型
    - 策略网络（Policy）：正在训练的策略，用于生成回答
    - 参考模型（Reference）：冻结的参考策略，用于计算 KL 散度
    """
    def __init__(self, model_name):
        super().__init__()
        # 加载预训练语言模型作为策略网络
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.policy_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # 创建参考模型（冻结参数）
        self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reference_model.eval()  # 设置为评估模式
        for param in self.reference_model.parameters():
            param.requires_grad = False  # 冻结参考模型参数
        
        if USE_GPU:
            self.to("cuda")
            self.reference_model.to("cuda")
    
    def forward(self, input_ids, attention_mask=None, use_amp=False):
        """策略网络前向传播，返回 logits"""
        device_type = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=use_amp and USE_GPU):
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
        return logits
    
    def get_reference_logits(self, input_ids, attention_mask=None, use_amp=False):
        """参考模型前向传播，返回 logits（不计算梯度）"""
        device_type = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            with autocast(device_type=device_type, enabled=use_amp and USE_GPU):
                outputs = self.reference_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
        return logits
    
    def sample_actions(self, input_ids, attention_mask=None, temperature=1.0):
        """
        根据当前策略采样动作（token）
        返回动作和对数概率
        """
        logits = self.forward(input_ids, attention_mask, use_amp=False)
        
        # 对最后一个位置的 logits 采样
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        dist = Categorical(probs)
        
        # 采样动作
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(self, input_ids, actions, attention_mask=None, use_amp=False):
        """
        评估给定动作在当前策略下的对数概率
        同时返回参考模型下的对数概率（用于计算 KL 散度）
        """
        # 当前策略的 logits
        logits = self.forward(input_ids, attention_mask, use_amp=use_amp)
        next_token_logits = logits[:, -1, :]
        
        # 使用 log_softmax 数值更稳定
        log_probs_all = torch.log_softmax(next_token_logits, dim=-1)
        probs = torch.exp(log_probs_all)
        
        # 处理数值不稳定性
        probs = probs.clamp(min=1e-10, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        # 参考模型的 logits
        ref_logits = self.get_reference_logits(input_ids, attention_mask, use_amp=use_amp)
        ref_next_token_logits = ref_logits[:, -1, :]
        ref_log_probs_all = torch.log_softmax(ref_next_token_logits, dim=-1)
        ref_probs = torch.exp(ref_log_probs_all)
        ref_probs = ref_probs.clamp(min=1e-10, max=1.0)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        
        ref_dist = Categorical(ref_probs)
        ref_log_probs = ref_dist.log_prob(actions)
        
        return log_probs, ref_log_probs


def compute_rewards(completions, ground_truth):
    """
    奖励函数：检查模型输出中 \boxed{} 内的答案是否正确
    正确返回 1.0，错误返回 0.0
    """
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def compute_grpo_advantages(rewards_group):
    """
    计算 GRPO 优势函数
    
    GRPO 的核心：使用组内相对奖励作为优势
    A_i = (r_i - mean(r_group)) / (std(r_group) + epsilon)
    
    这样不需要价值网络，直接使用组内比较来估计优势
    """
    rewards_tensor = torch.tensor(rewards_group, dtype=torch.float32)
    mean_reward = rewards_tensor.mean()
    std_reward = rewards_tensor.std()
    
    # 计算相对优势
    advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-8)
    
    return advantages.tolist()


def compute_kl_divergence(log_probs, ref_log_probs):
    """
    计算 KL 散度：KL(参考策略 || 当前策略)
    
    使用近似公式：KL ≈ ref_log_probs - log_probs
    这是 GRPO 中常用的近似方法
    """
    return ref_log_probs - log_probs


def grpo_loss(old_log_probs, log_probs, ref_log_probs, advantages, epsilon=0.2, beta=0.01):
    """
    GRPO 损失函数
    
    包含两部分：
    1. PPO-clip 策略损失：限制策略更新幅度
    2. KL 散度惩罚：防止策略偏离参考模型太远
    
    目标函数：
    L = E[ min(r * A, clip(r, 1-eps, 1+eps) * A) - beta * KL ]
    
    其中 r = exp(log_probs - old_log_probs) 是概率比
    """
    # 计算概率比
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 将 advantages 转换为与 ratio 相同的设备
    advantages = torch.tensor(advantages, dtype=torch.float32)
    if log_probs.is_cuda:
        advantages = advantages.to(log_probs.device)
    
    # PPO-clip 策略损失
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # KL 散度惩罚
    kl_divergence = compute_kl_divergence(log_probs, ref_log_probs)
    kl_penalty = beta * kl_divergence.mean()
    
    # 总损失
    loss = policy_loss + kl_penalty
    
    return loss, policy_loss.item(), kl_penalty.item()


def build_dataset(data_num=50):
    """构建训练数据集"""
    # 加载数据集并获取训练集
    local_gsm8k_path = r"C:\Users\曾\Desktop\84Post-training of LLMs\L7\gsm8k"
    dataset_dict = load_dataset(local_gsm8k_path, "main")
    dataset = dataset_dict["train"]  # 获取训练集
    
    def post_process(example):
        match = re.search(r"####\s*(-?\d+)", example["answer"])
        example["ground_truth"] = match.group(1) if match else None
        example["query"] = f"{SYSTEM_PROMPT}\n\nQuestion: {example['question']}"
        return example
    
    dataset = dataset.map(post_process)
    dataset = dataset.remove_columns(["question", "answer"])
    dataset = dataset.select(range(data_num))
    return dataset


def collect_group_rollouts(agent, query, ground_truth, group_size=4, max_new_tokens=50, temperature=1.0):
    """
    收集组内经验（Group Rollouts）
    
    这是 GRPO 的核心步骤：
    对每个问题采样 G 个回答，形成一组样本
    
    返回：
    - responses: G 个回答文本
    - all_log_probs: 每个回答的 token 级别对数概率
    - all_ref_log_probs: 参考模型下的对数概率
    - rewards: G 个回答的奖励
    """
    # 编码输入
    input_ids = agent.tokenizer.encode(query, return_tensors="pt")
    if USE_GPU:
        input_ids = input_ids.to("cuda")
    
    group_memories = []
    
    for g in range(group_size):
        # 逐步生成响应
        generated_ids = input_ids.clone()
        log_probs_list = []
        ref_log_probs_list = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # 采样动作
                action, log_prob = agent.sample_actions(generated_ids, temperature=temperature)
                
                # 获取参考模型的对数概率
                ref_log_prob, _ = agent.evaluate_actions(generated_ids, action)
                ref_log_probs_list.append(ref_log_prob.item())
            
            log_probs_list.append(log_prob.item())
            
            # 添加到序列
            generated_ids = torch.cat([generated_ids, action.unsqueeze(0)], dim=1)
            
            # 检查是否生成了结束符
            if action.item() == agent.tokenizer.eos_token_id:
                break
        
        # 解码生成的响应
        response = agent.tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        group_memories.append({
            "generated_ids": generated_ids,
            "log_probs": log_probs_list,
            "ref_log_probs": ref_log_probs_list,
            "response": response,
        })
    
    # 计算奖励
    responses = [mem["response"] for mem in group_memories]
    rewards = compute_rewards(responses, [ground_truth] * group_size)
    
    # 将奖励添加到每个记忆中
    for i, mem in enumerate(group_memories):
        mem["reward"] = rewards[i]
    
    return group_memories, rewards


def update_policy_grpo(agent, group_memories, advantages, optimizer, scaler, 
                       ppo_epochs=4, epsilon=0.2, beta=0.01):
    """
    使用 GRPO 更新策略
    
    步骤：
    1. 对每个组内样本计算新的对数概率
    2. 计算 GRPO 损失（PPO-clip + KL 惩罚）
    3. 反向传播更新策略
    """
    total_loss = 0
    total_policy_loss = 0
    total_kl_loss = 0
    
    for mem_idx, memory in enumerate(group_memories):
        # 准备数据
        generated_ids = memory["generated_ids"]
        old_log_probs = torch.tensor(memory["log_probs"])
        old_ref_log_probs = torch.tensor(memory["ref_log_probs"])
        advantage = advantages[mem_idx]
        
        if USE_GPU:
            old_log_probs = old_log_probs.to("cuda")
            old_ref_log_probs = old_ref_log_probs.to("cuda")
            generated_ids = generated_ids.to("cuda")
        
        # 获取动作序列
        actions = generated_ids[0][1:]  # 去掉第一个输入 token
        
        # PPO 更新
        for _ in range(ppo_epochs):
            new_log_probs_list = []
            new_ref_log_probs_list = []
            
            for t in range(len(actions)):
                input_ids_t = generated_ids[:, :t+1]
                action_t = actions[t:t+1]
                
                # 评估动作
                new_log_prob, new_ref_log_prob = agent.evaluate_actions(
                    input_ids_t, action_t, use_amp=USE_AMP
                )
                new_log_probs_list.append(new_log_prob)
                new_ref_log_probs_list.append(new_ref_log_prob)
            
            new_log_probs = torch.stack(new_log_probs_list)
            new_ref_log_probs = torch.stack(new_ref_log_probs_list)
            
            # 计算 GRPO 损失
            loss, policy_loss, kl_loss = grpo_loss(
                old_log_probs, new_log_probs, new_ref_log_probs, 
                [advantage] * len(actions), epsilon, beta
            )
            
            # 反向传播
            optimizer.zero_grad()
            
            if USE_AMP and USE_GPU:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss
            total_kl_loss += kl_loss
    
    n_updates = len(group_memories) * ppo_epochs
    return total_loss / n_updates, total_policy_loss / n_updates, total_kl_loss / n_updates


def train_grpo():
    """主训练函数"""
    model_name = r"C:\Users\曾\Desktop\84Post-training of LLMs\L5\models\Qwen2___5-0___5B-Instruct"
    num_iterations = 2  # 总迭代次数
    steps_per_iter = 2  # 每次迭代处理的问题数
    group_size = 4  # 每个问题的采样数（GRPO 核心参数）
    max_new_tokens = 100  # 每个回答的最大 token 数
    learning_rate = 5e-6
    ppo_epochs = 2  # 每个样本的更新次数
    epsilon = 0.2  # PPO clip 参数
    beta = 0.01  # KL 散度惩罚系数
    temperature = 1.0  # 采样温度
    
    # 检查 GPU 可用性
    if USE_GPU and torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用 GPU: {device} ({memory:.1f} GB)")
        print(f"混合精度训练: {'启用' if USE_AMP else '禁用'}")
    else:
        print("警告：GPU 不可用，将使用 CPU（速度会很慢）")
    
    print("=" * 60)
    print("GRPO 训练开始")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"迭代次数: {num_iterations}")
    print(f"每步问题数: {steps_per_iter}")
    print(f"组大小 (G): {group_size}")
    print(f"学习率: {learning_rate}")
    print(f"KL 惩罚系数 (beta): {beta}")
    print("=" * 60)
    
    # 初始化
    agent = GRPOAgent(model_name)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    scaler = GradScaler() if USE_AMP and USE_GPU else None
    dataset = build_dataset(data_num=steps_per_iter * num_iterations)
    
    # 训练循环
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        start_idx = iteration * steps_per_iter
        end_idx = start_idx + steps_per_iter
        iter_dataset = dataset.select(range(start_idx, end_idx))
        
        all_rewards = []
        
        for step in range(steps_per_iter):
            query = iter_dataset[step]["query"]
            ground_truth = iter_dataset[step]["ground_truth"]
            
            print(f"\n[步骤 {step+1}/{steps_per_iter}] 问题: {query[:80]}...")
            
            # 1. 收集组内经验（Group Rollout）
            print(f"  采样 {group_size} 个回答...")
            group_memories, rewards = collect_group_rollouts(
                agent, query, ground_truth, 
                group_size=group_size, 
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            all_rewards.extend(rewards)
            
            # 显示组内结果
            print(f"  组内奖励: {rewards}")
            print(f"  组内平均奖励: {np.mean(rewards):.4f}")
            
            for g, mem in enumerate(group_memories):
                print(f"    回答 {g+1}: {mem['response'][:80]}... (奖励: {mem['reward']})")
            
            # 2. 计算 GRPO 优势函数
            advantages = compute_grpo_advantages(rewards)
            print(f"  计算优势: {[f'{a:.3f}' for a in advantages]}")
            
            # 3. 更新策略
            print("  更新策略...")
            loss, policy_loss, kl_loss = update_policy_grpo(
                agent, group_memories, advantages, optimizer, scaler,
                ppo_epochs=ppo_epochs, epsilon=epsilon, beta=beta
            )
            print(f"  损失: {loss:.4f} (策略: {policy_loss:.4f}, KL: {kl_loss:.4f})")
            
            # 清理显存缓存
            torch.cuda.empty_cache()
        
        # 显示本轮统计
        print(f"\n{'='*60}")
        print(f"迭代 {iteration + 1} 完成")
        print(f"平均奖励: {np.mean(all_rewards):.4f}")
        print(f"{'='*60}")
    
    # 保存模型
    output_dir = "./grpo_scratch_output"
    agent.policy_model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"\n{'='*60}")
    print(f"训练完成！模型已保存到: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_grpo()
