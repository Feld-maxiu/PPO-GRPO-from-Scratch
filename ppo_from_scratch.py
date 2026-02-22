"""
PPO (Proximal Policy Optimization) 从头实现
用于理解 PPO 算法的每个步骤

PPO 核心思想：
1. 收集经验：使用当前策略生成响应，并计算奖励
2. 计算优势函数：估计每个动作比平均水平好多少
3. 更新策略：使用裁剪的目标函数来限制策略更新幅度
4. 更新价值函数：让价值网络更准确地估计期望回报
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
    "You must follow these rules strictly:\n"
    "1. Do NOT show any calculation steps\n"
    "2. Do NOT explain your reasoning\n"
    "3. ONLY output the final answer in this exact format: The answer is \\boxed{YOUR_ANSWER}\n"
    "4. Your entire response should be just one sentence: 'The answer is \\boxed{...}'"
)


class ValueHead(nn.Module):
    """
    价值网络头：为每个输入序列预测一个标量价值
    用于估计当前状态的期望回报
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, hidden_states):
        # 取最后一个 token 的隐藏状态来预测价值
        return self.value_head(hidden_states[:, -1, :]).squeeze(-1)


class PPOAgent(nn.Module):
    """
    PPO 智能体：包含策略网络和价值网络
    - 策略网络（Policy）：决定生成什么 token
    - 价值网络（Value）：评估当前状态的价值
    """
    def __init__(self, model_name):
        super().__init__()
        # 加载预训练语言模型作为策略网络
        self.policy_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.policy_model.config.pad_token_id = self.tokenizer.eos_token_id
        
        # 添加价值网络头
        self.value_head = ValueHead(self.policy_model.config.hidden_size)
        
        if USE_GPU:
            self.to("cuda")
    
    def forward(self, input_ids, attention_mask=None, use_amp=False):
        """前向传播，返回 logits 和价值估计"""
        # 使用 autocast 进行混合精度计算
        device_type = 'cuda' if USE_GPU and torch.cuda.is_available() else 'cpu'
        with autocast(device_type=device_type, enabled=use_amp and USE_GPU):
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            logits = outputs.logits
            values = self.value_head(outputs.hidden_states[-1])
        return logits, values
    
    def get_action_and_value(self, input_ids, attention_mask=None):
        """
        根据当前策略选择动作（token），并返回该动作的对数概率和价值
        """
        logits, values = self.forward(input_ids, attention_mask, use_amp=False)
        
        # 对最后一个位置的 logits 采样
        next_token_logits = logits[:, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        dist = Categorical(probs)
        
        # 采样动作
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob, values
    
    def evaluate_actions(self, input_ids, actions, attention_mask=None, use_amp=False):
        """
        评估给定动作的对数概率和当前状态价值
        用于 PPO 的更新阶段
        """
        logits, values = self.forward(input_ids, attention_mask, use_amp=use_amp)
        next_token_logits = logits[:, -1, :]
        
        # 使用 log_softmax 而不是 softmax，数值更稳定
        log_probs_all = torch.log_softmax(next_token_logits, dim=-1)
        probs = torch.exp(log_probs_all)
        
        # 处理数值不稳定性：确保概率和为1，避免出现无效值
        probs = probs.clamp(min=1e-10, max=1.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values, entropy


def compute_rewards(completions, ground_truth):
    """
    奖励函数：检查模型输出中 \boxed{} 内的答案是否正确
    正确返回 1.0，错误返回 0.0
    """
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


def compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95):
    """
    GAE (Generalized Advantage Estimation) 计算优势函数
    
    优势函数 A(s,a) = Q(s,a) - V(s)
    表示采取动作 a 比平均水平好多少
    
    GAE 公式：
    A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
    其中 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
    advantages = []
    gae = 0
    
    # 从后向前计算
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0  # 终止状态的价值为 0
        else:
            next_value = values[t + 1]
        
        # TD 误差
        delta = rewards[t] + gamma * next_value - values[t]
        
        # GAE 累积
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    return torch.tensor(advantages, dtype=torch.float32)


def ppo_loss(old_log_probs, log_probs, advantages, epsilon=0.2):
    """
    PPO 裁剪目标函数
    
    目标：最大化以下目标函数
    L^{CLIP}(theta) = E[ min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t) ]
    
    其中 r_t = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) 是概率比
    
    裁剪的作用：防止策略更新过大，保持训练的稳定性
    """
    # 计算概率比
    ratio = torch.exp(log_probs - old_log_probs)
    
    # 未裁剪的目标
    surr1 = ratio * advantages
    
    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # 取最小值（ pessimistic bound ）
    loss = -torch.min(surr1, surr2).mean()
    
    return loss


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


def collect_rollouts(agent, dataset, max_new_tokens=50):
    """
    收集经验（Rollouts）
    
    这是 PPO 的第一步：使用当前策略与环境交互，收集 (s, a, r) 轨迹
    """
    memories = []
    
    for i in range(len(dataset)):
        query = dataset[i]["query"]
        ground_truth = dataset[i]["ground_truth"]
        
        # 编码输入
        input_ids = agent.tokenizer.encode(query, return_tensors="pt")
        if USE_GPU:
            input_ids = input_ids.to("cuda")
        
        # 逐步生成响应
        generated_ids = input_ids.clone()
        log_probs_list = []
        values_list = []
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                action, log_prob, value = agent.get_action_and_value(generated_ids)
            
            # 记录动作信息
            log_probs_list.append(log_prob.item())
            values_list.append(value.item())
            
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
        
        # 计算奖励
        reward = compute_rewards([response], [ground_truth])[0]
        
        memories.append({
            "query": query,
            "response": response,
            "ground_truth": ground_truth,
            "generated_ids": generated_ids,
            "log_probs": log_probs_list,
            "values": values_list,
            "reward": reward,
        })
    
    return memories


def update_policy(agent, memories, optimizer, scaler, ppo_epochs=4, epsilon=0.2,
                  value_coef=0.5, entropy_coef=0.01):
    """
    使用收集的经验更新策略
    
    这是 PPO 的核心更新步骤：
    1. 计算优势函数
    2. 多次迭代更新策略（使用旧策略的数据）
    3. 同时更新价值网络
    """
    total_loss = 0
    
    for memory in memories:
        # 准备数据
        generated_ids = memory["generated_ids"]
        old_log_probs = torch.tensor(memory["log_probs"])
        old_values = torch.tensor(memory["values"])
        reward = memory["reward"]
        
        if USE_GPU:
            old_log_probs = old_log_probs.to("cuda")
            old_values = old_values.to("cuda")
            generated_ids = generated_ids.to("cuda")
        
        # 构建奖励序列（每个 token 的奖励，只有最后一个是实际奖励）
        rewards_seq = [0.0] * (len(old_log_probs) - 1) + [reward]
        
        # 使用 GAE 计算优势函数
        advantages = compute_gae(rewards_seq, old_values.tolist(), gamma=0.99, gae_lambda=0.95)
        if USE_GPU:
            advantages = advantages.to("cuda")
        
        # 计算回报：returns = advantages + values
        returns = advantages + old_values
        
        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO 更新
        for _ in range(ppo_epochs):
            # 重新评估动作
            actions = generated_ids[0][1:]  # 去掉第一个输入 token
            new_log_probs_list = []
            new_values_list = []
            entropy_list = []
            
            for t in range(len(actions)):
                input_ids_t = generated_ids[:, :t+1]
                action_t = actions[t:t+1]
                
                # 使用混合精度评估动作
                new_log_prob, new_value, entropy = agent.evaluate_actions(
                    input_ids_t, action_t, use_amp=USE_AMP
                )
                new_log_probs_list.append(new_log_prob)
                new_values_list.append(new_value)
                entropy_list.append(entropy)
            
            new_log_probs = torch.stack(new_log_probs_list)
            new_values = torch.stack(new_values_list)
            entropy = torch.stack(entropy_list).mean()
            
            # 计算 PPO 损失
            policy_loss = ppo_loss(
                old_log_probs, new_log_probs, advantages, epsilon
            )
            
            # 价值函数损失（MSE）
            value_loss = nn.MSELoss()(new_values, returns)
            
            # 总损失
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # 反向传播（使用混合精度）
            optimizer.zero_grad()
            
            if USE_AMP and USE_GPU:
                # 使用 GradScaler 进行混合精度反向传播
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
    
    return total_loss / len(memories)


def train_ppo():
    """主训练函数"""
    model_name = r"C:\Users\曾\Desktop\84Post-training of LLMs\L5\models\Qwen2___5-0___5B-Instruct"
    num_iterations = 2  # 总迭代次数
    steps_per_iter = 2  
    max_new_tokens = 10 
    learning_rate = 5e-6
    ppo_epochs = 2  
    
    # 检查 GPU 可用性
    if USE_GPU and torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"使用 GPU: {device} ({memory:.1f} GB)")
        print(f"混合精度训练: {'启用' if USE_AMP else '禁用'}")
    else:
        print("警告：GPU 不可用，将使用 CPU（速度会很慢）")
    
    print("=" * 60)
    print("PPO 训练开始")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"迭代次数: {num_iterations}")
    print(f"每步经验数: {steps_per_iter}")
    print(f"学习率: {learning_rate}")
    print("=" * 60)
    
    # 初始化
    agent = PPOAgent(model_name)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
    scaler = GradScaler() if USE_AMP and USE_GPU else None  # 混合精度训练用的 GradScaler
    dataset = build_dataset(data_num=steps_per_iter * num_iterations)
    
    # 训练循环
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"迭代 {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # 1. 收集经验（Rollout）
        print("\n[步骤 1] 收集经验...")
        start_idx = iteration * steps_per_iter
        end_idx = start_idx + steps_per_iter
        iter_dataset = dataset.select(range(start_idx, end_idx))
        
        memories = collect_rollouts(agent, iter_dataset, max_new_tokens=max_new_tokens)
        
        # 显示收集到的经验
        avg_reward = np.mean([m["reward"] for m in memories])
        print(f"平均奖励: {avg_reward:.4f}")
        
        for i, mem in enumerate(memories):
            print(f"\n样本 {i+1}:")
            print(f"  问题: {mem['query'][:100]}...")
            print(f"  回答: {mem['response'][:100]}...")
            print(f"  正确答案: {mem['ground_truth']}")
            print(f"  奖励: {mem['reward']}")
        
        # 2. 更新策略
        print("\n[步骤 2] 更新策略...")
        loss = update_policy(agent, memories, optimizer, scaler, ppo_epochs=ppo_epochs)
        print(f"损失: {loss:.4f}")
        
        # 清理显存缓存
        torch.cuda.empty_cache()
    
    # 保存模型
    output_dir = "./ppo_scratch_output"
    agent.policy_model.save_pretrained(output_dir)
    agent.tokenizer.save_pretrained(output_dir)
    print(f"\n{'='*60}")
    print(f"训练完成！模型已保存到: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train_ppo()
