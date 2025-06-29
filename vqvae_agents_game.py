import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import random
import torch.distributions
import os
from datetime import datetime
import json 

# =======================
# AIMDictionary (修改以記錄非 AIM 相關的遊戲數據)
# =======================
class AIMDictionary:
    def __init__(self, filename="game_log.json"):
        self.log_data = []
        self.filename = filename
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    self.log_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode existing {self.filename}. Starting with empty log.")
                self.log_data = []

    def add_entry(self, round_num, label, action_A, action_B, reward_A, reward_B, joint_reward, context=""):
        entry = {
            "round": round_num,
            "label": label,
            "agent_A_action": action_A,
            "agent_B_action": action_B,
            "reward_A": reward_A,
            "reward_B": reward_B,
            "joint_reward": joint_reward,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.log_data.append(entry)

    def save(self):
        with open(self.filename, 'w') as f:
            json.dump(self.log_data, f, indent=4)
        print(f"Game log saved to {self.filename}")

# =======================
# VQ-VAE (保持不變)
# =======================
class Encoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, D)
        )

    def forward(self, x):
        return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z_q):
        return self.dec(z_q).view(-1, 1, 28, 28)

class VectorQuantizer(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1/K, 1/K)

    def forward(self, z_e):
        dist = torch.cdist(z_e, self.codebook.weight)
        encoding_inds = torch.argmin(dist, dim=1)
        z_q = self.codebook(encoding_inds)
        return z_q, encoding_inds

class VQVAE(nn.Module):
    def __init__(self, K=16, D=64):
        super().__init__()
        self.encoder = Encoder(D)
        self.quantizer = VectorQuantizer(K, D)
        self.decoder = Decoder(D)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, encoding_inds = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_e, z_q, encoding_inds

# =======================
# Agents (修改為直接輸出 C/D 行動，並集成 VQ-VAE 情境)
# =======================
class AgentA(nn.Module):
    def __init__(self, vqvae): 
        super().__init__()
        self.vqvae = vqvae
        self.label_embed = nn.Embedding(10, 8)
        # 行動嵌入：0 for C, 1 for D (Cooperate/Defect)
        self.action_embed = nn.Embedding(2, 8) 

        # 策略網路輸入：z_e (圖像編碼) + label_feat (標籤嵌入)
        policy_input_dim = vqvae.encoder.enc[-1].out_features + self.label_embed.embedding_dim

        # Actor (Policy Network): 輸出 C/D 的 logits (2 個)
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 2 個 logits: C 或 D
        )
        
        # Critic (Value Network): 接收 AgentA 自己的情境和 AgentB 的行動
        # Critic 輸入: z_e + label_feat + opponent_action_embed
        critic_input_dim = policy_input_dim + self.action_embed.embedding_dim
        self.value_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 1) # 輸出聯合獎勵的價值
        )
        
        # 意圖預測器 A: 根據自己的情境預測自己的行動
        # 輸入：z_e + 標籤嵌入
        # 輸出：2 個 logits (Cooperate 或 Defect)
        self.intent_predictor_A = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # 2 個類別: C 或 D
        )

    # forward 函數：為不同模式提供不同輸出
    # x: 原始圖像, label: 圖像標籤, opponent_action: 對手實際行動 (0 for C, 1 for D)
    def forward(self, x, label, opponent_action=None, mode='policy'):
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)
        combined_base_input = torch.cat([z_e, label_feat], dim=1) # 這是策略網路和意圖預測的基本輸入

        if mode == 'policy': # 輸出行動的 logits 和狀態價值
            action_logits = self.policy_net(combined_base_input)
            
            # Critic 輸入需要對手的實際行動。在 AgentA 第一次調用 policy 模式時，
            # B_sampled_action 還未生成，這裡需要一個預設值或在 multi_agent_game 中處理
            # 在 multi_agent_game 中，我們會在 B_sampled_action 生成後重新調用 AgentA 獲取價值。
            # 因此這裡的 opponent_action 在計算 A_value 時不會是 None
            if opponent_action is None:
                # 僅為確保邏輯完整性，實際使用時應確保不為 None
                dummy_action = torch.tensor([0]).to(label.device) # 預設為 C
                embedded_opponent_action = self.action_embed(dummy_action)
            else:
                embedded_opponent_action = self.action_embed(opponent_action)
            
            combined_critic_input = torch.cat([combined_base_input, embedded_opponent_action], dim=1)
            value = self.value_net(combined_critic_input)
            
            return action_logits, value.squeeze(-1)
            
        elif mode == 'predict_own_intent': # 預測自己的意圖
            # 輸入就是 combined_base_input (z_e + label_feat)
            return self.intent_predictor_A(combined_base_input) # 返回 logits
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentA.")


class AgentB(nn.Module):
    def __init__(self, vqvae): 
        super().__init__()
        self.vqvae = vqvae
        self.label_embed = nn.Embedding(10, 8)
        self.action_embed = nn.Embedding(2, 8) # 0 for C, 1 for D

        # 策略網路輸入：z_e (圖像編碼) + label_feat (標籤嵌入) + received_opponent_action_embed (對手行動嵌入)
        policy_input_dim = (self.vqvae.encoder.enc[-1].out_features + 
                            self.label_embed.embedding_dim + 
                            self.action_embed.embedding_dim) 

        # Actor (Policy Network) 輸出 C/D 的 logits (2 個)
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2) # 2 個 logits: C 或 D
        )

        # 意圖解碼器 B: 根據情境和對手的行動解碼對手意圖 (預測對手的行動)
        # 輸入：z_e + 標籤嵌入 + 對手行動嵌入
        self.intent_decoder_B = nn.Sequential(
            nn.Linear(policy_input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 2) # 2 個類別: C 或 D
        )
    
    # forward 函數：接收原始圖像 x, 標籤 label, 和對手行動 opponent_action
    def forward(self, x, label, opponent_action, mode='policy'): 
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)
        embedded_opponent_action = self.action_embed(opponent_action)

        # 將所有輸入拼接：原始圖像編碼 z_e, 標籤, 對手行動嵌入
        combined_input_for_nets = torch.cat([z_e, label_feat, embedded_opponent_action], dim=1) 

        if mode == 'policy': # AgentB 只輸出行動的 logits
            action_logits = self.policy_net(combined_input_for_nets) 
            return action_logits
            
        elif mode == 'decode_opponent_intent': # 解碼對手意圖
            return self.intent_decoder_B(combined_input_for_nets) # 返回 logits
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentB.")

# =======================
# Game Logic & RL Components
# =======================

# 情境化獎勵
def payoff(action_A, action_B, image_label, current_round):
    rA, rB = 0, 0

    # A. 基礎獎勵
    if action_A == 'C' and action_B == 'C': 
        rA, rB = 3, 3 
    elif action_A == 'C' and action_B == 'D': 
        rA, rB = -1, 5 
    elif action_A == 'D' and action_B == 'C': 
        rA, rB = 5, -1
    elif action_A == 'D' and action_B == 'D': 
        rA, rB = 0, 0 

    # B. 情境獎勵
    if image_label % 2 == 0:  # 偶數：額外鼓勵合作
        if action_A == 'C' and action_B == 'C':
            rA += 2 # 從 (3,3) 變為 (5,5)
            rB += 2
        elif (action_A == 'C' and action_B == 'D'):
            rA -= 1 # 合作方懲罰
        elif (action_A == 'D' and action_B == 'C'):
            rB -= 1 # 合作方懲罰
    else:  # 奇數：輕微懲罰單方面合作，但 C,C 仍是最佳 (C,C 獎勵仍是基礎獎勵 3,3)
        # 這裡的獎勵計算與偶數情況相同，只是 C,C 沒有額外獎勵
        # 所以奇數 C,C 仍是 (3,3)
        if (action_A == 'C' and action_B == 'D'):
            rA -= 1 # 合作方懲罰
        elif (action_A == 'D' and action_B == 'C'):
            rB -= 1 # 合作方懲罰

    return rA, rB

def train_vqvae(epochs, K_val, D_val):
    transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)

    vqvae = VQVAE(K=K_val, D=D_val)
    optimizer = optim.Adam(vqvae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("\n--- Training VQ-VAE ---")
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (x, _) in enumerate(loader):
            x_hat, z_e, z_q, _ = vqvae(x)
            
            recon_loss = loss_fn(x_hat, x)
            commit_loss = ((z_e - z_q.detach()) ** 2).mean()
            codebook_loss = ((z_q - z_e.detach()) ** 2).mean()

            loss = recon_loss + 0.25 * commit_loss + 1.0 * codebook_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        print(f'VQ-VAE Epoch {epoch+1}/{epochs}, Avg Loss: {epoch_loss / len(loader):.4f}')
    return vqvae

def multi_agent_game(vqvae, aim_dict, rounds=5, 
                     reflection_strategy='intent_alignment', reflection_coeff=0.1, gamma_rl=0.99, entropy_coeff=0.01): 
 
# *** 凍結 VQ-VAE 的參數 ***
    for param in vqvae.parameters():
        param.requires_grad = False # 設置為 False，使其在 Agent 訓練中不計算梯度

    agentA = AgentA(vqvae) 
    agentB = AgentB(vqvae) 

    optimizer_A = optim.Adam(list(agentA.parameters()), lr=1e-4) 
    optimizer_B = optim.Adam(list(agentB.parameters()), lr=1e-4)

    scheduler_A = torch.optim.lr_scheduler.ExponentialLR(optimizer_A, gamma=0.9995) 
    scheduler_B = torch.optim.lr_scheduler.ExponentialLR(optimizer_B, gamma=0.9995) 

    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    A_rewards_history, B_rewards_history, Joint_rewards_history = [], [], [] 

    all_labels = torch.arange(10).repeat(rounds // 10 + 1)[:rounds].tolist()
    random.shuffle(all_labels)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_data):
        label_to_indices[label].append(idx)

    value_loss_fn = nn.MSELoss() 
    intent_loss_fn = nn.CrossEntropyLoss() 

    initial_entropy_coeff = entropy_coeff
    entropy_decay_rate = 0.9999 

    print(f"\n--- Starting Multi-Agent Contextual Prisoner's Dilemma Game (Strategy: {reflection_strategy}) ---")
    for i in range(rounds):
        current_label = all_labels[i] 
        matching_indices = label_to_indices[current_label]
        if not matching_indices:
            # Fallback if no images found for a label (shouldn't happen with MNIST fully downloaded)
            idx = random.randint(0, len(test_data)-1)
            x, _ = test_data[idx] 
        else:
            idx = random.choice(matching_indices)
            x, _ = test_data[idx]

        x = x.unsqueeze(0) 
        current_label_tensor = torch.tensor([current_label]) 

        # 1. Agent 產生行動 (Actor 部分)
        # AgentA 策略輸出。這裡先給一個 dummy opponent_action=0，以便獲取 action_logits。
        # 實際的 A_value 計算會在 B_sampled_action 生成後進行。
        A_action_logits_policy, _ = agentA(x, current_label_tensor, mode='policy', opponent_action=torch.tensor([0])) 
        A_dist = torch.distributions.Categorical(logits=A_action_logits_policy) 
        A_sampled_action = A_dist.sample() # 0 (C) 或 1 (D)
        A_log_probs = A_dist.log_prob(A_sampled_action) 
        A_entropy = A_dist.entropy() 

        # AgentB 策略輸出，接收 AgentA 的行動
        B_action_logits_policy = agentB(x, current_label_tensor, A_sampled_action, mode='policy') 
        B_dist = torch.distributions.Categorical(logits=B_action_logits_policy)
        B_sampled_action = B_dist.sample() # 0 (C) 或 1 (D)
        B_log_probs = B_dist.log_prob(B_sampled_action)
        B_entropy = B_dist.entropy()

        # 將數字行動轉換為 C/D 字串
        A_action_human_interp = 'C' if A_sampled_action.item() == 0 else 'D'
        B_action_human_interp = 'C' if B_sampled_action.item() == 0 else 'D'

        # 2. 計算獎勵
        A_reward_indiv, B_reward_indiv = payoff(
            A_action_human_interp, B_action_human_interp, current_label, i + 1 
        )
        joint_reward = A_reward_indiv + B_reward_indiv 
        
        # 3. 計算中心化 Critic 的價值 (在 AgentA 中)
        # 重新調用 AgentA 的 forward 模式 'policy' 來計算價值，這次傳遞實際的 B_sampled_action
        _, A_value = agentA(x, current_label_tensor, mode='policy', opponent_action=B_sampled_action)

        # 4. 計算核心 A2C 損失
        # Agent A 損失 (Actor-Critic)
        A_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach() 
        loss_A_policy = - (A_log_probs * A_advantage.to(A_value.device)) 
        loss_A_value = value_loss_fn(A_value, torch.tensor([joint_reward], dtype=torch.float32).to(A_value.device)) 
        
        current_entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** i)
        loss_A = loss_A_policy + 0.5 * loss_A_value - current_entropy_coeff * A_entropy

        # Agent B 損失 (策略損失基於共同 Critic 的 Advantage)
        B_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach() 
        loss_B_policy = - (B_log_probs * B_advantage.to(A_value.device))
        
        loss_B = loss_B_policy - current_entropy_coeff * B_entropy

        # 5. 意圖溝通對齊損失
        if reflection_strategy == 'intent_alignment':
            target_A_action_idx = A_sampled_action # 0 或 1
            
            # Agent A 預測自己的意圖 (輸入是情境，輸出是自己的行動預測)
            predicted_A_intent_logits = agentA(x, current_label_tensor, mode='predict_own_intent')
            loss_A_own_intent = intent_loss_fn(predicted_A_intent_logits, target_A_action_idx)
            loss_A += reflection_coeff * loss_A_own_intent 

            # Agent B 解碼對手意圖 (輸入是情境和對手行動，輸出是對手行動的預測)
            predicted_B_decoded_intent_logits = agentB(x, current_label_tensor, A_sampled_action, mode='decode_opponent_intent')
            loss_B_decode_intent = intent_loss_fn(predicted_B_decoded_intent_logits, target_A_action_idx) # B 預測 A 的行動
            loss_B += reflection_coeff * loss_B_decode_intent 
        
        # 清除梯度並執行反向傳播和優化
        optimizer_A.zero_grad()
        loss_A.backward()
        optimizer_A.step()

        optimizer_B.zero_grad()
        loss_B.backward()
        optimizer_B.step()

        scheduler_A.step()
        scheduler_B.step()

        # 記錄獎勵
        A_rewards_history.append(A_reward_indiv)
        B_rewards_history.append(B_reward_indiv)
        Joint_rewards_history.append(joint_reward) 
        
        # 將遊戲數據添加到 AIMDictionary (現在是通用日誌)
        aim_dict.add_entry(i+1, current_label, A_action_human_interp, B_action_human_interp, 
                           A_reward_indiv, B_reward_indiv, joint_reward, f"Round {i+1} Context")

        print(f'Round {i+1}/{rounds}: '
              f'Label={current_label} | '
              f'Agent A Action={A_action_human_interp} | '
              f'Agent B Action={B_action_human_interp} | '
              f'Reward A={A_reward_indiv:.2f}, B={B_reward_indiv:.2f} | '
              f'Joint Reward={joint_reward:.2f} | '
              f'Avg A={sum(A_rewards_history) / len(A_rewards_history):.2f}, '
              f'Avg B={sum(B_rewards_history) / len(B_rewards_history):.2f}, '
              f'Avg Joint={sum(Joint_rewards_history) / len(Joint_rewards_history):.2f}')

    return A_rewards_history, B_rewards_history, Joint_rewards_history

def visualize(A_rewards, B_rewards, Joint_rewards, strategy_name): 
     # 僅繪製聯合獎勵
    plt.plot(Joint_rewards, label='Joint Reward', alpha=0.7, linestyle='--', color='red') 
    plt.title(f'Payoff Over Time (Strategy: {strategy_name}) - Joint Reward Only') # 修改標題以反映只顯示聯合獎勵
 
   
    #plt.figure(figsize=(12, 7))
    #plt.plot(A_rewards, label='A Reward', alpha=0.7)
    #plt.plot(B_rewards, label='B Reward', alpha=0.7)
    
    #plt.title(f'Payoff Over Time (Strategy: {strategy_name})')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# =======================
# Main with CLI
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Agent Contextual Prisoner's Dilemma Game with Intent Alignment")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for VQ-VAE') 
    parser.add_argument('--rounds', type=int, default=10000, help='Number of multi-agent game rounds (more for RL)') 
    parser.add_argument('--K', type=int, default=32, help='Size of the VQ-VAE codebook. Still used for VQVAE.') 
    parser.add_argument('--D', type=int, default=64, help='Dimension of the VQ-VAE code vectors. Still used for VQVAE.')
    parser.add_argument('--reflection_strategy', type=str, default='intent_alignment', 
                        choices=['none', 'intent_alignment'], 
                        help='Reflection strategy to use: none, or intent_alignment (default).')
    parser.add_argument('--reflection_coeff', type=float, default=0.05, 
                        help='Coefficient for the reflection loss term. Adjust as needed.')
    parser.add_argument('--gamma_rl', type=float, default=0.99, help='Discount factor for RL rewards (gamma_rl in A2C).')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Initial coefficient for entropy regularization.') 
    args = parser.parse_args()

    aim_dict = AIMDictionary() # 使用修改後的 AIMDictionary
    vqvae = train_vqvae(args.epochs, args.K, args.D)

    A_rewards, B_rewards, Joint_rewards = multi_agent_game(vqvae, aim_dict, rounds=args.rounds,
                                            reflection_strategy=args.reflection_strategy,
                                            reflection_coeff=args.reflection_coeff,
                                            gamma_rl=args.gamma_rl, 
                                            entropy_coeff=args.entropy_coeff)
    
    aim_dict.save() 
    visualize(A_rewards, B_rewards, Joint_rewards, args.reflection_strategy)