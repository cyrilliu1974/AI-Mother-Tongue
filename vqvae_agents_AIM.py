import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from aim_dictionary_json import AIMDictionary
import argparse
import random
import torch.distributions
import os
from datetime import datetime
import json

# =======================
#This program implements "AI Mother Tongue: Self-Emergent Communication in MARL via Endogenous Symbol Systems" arXiv:2507.10566
# =======================

# =======================
# VQ-VAE Implementation
# Implements the Vector Quantized Variational Autoencoder (VQ-VAE) as described in the paper (Page 6, Section 3.1).
# The VQ-VAE transforms continuous input features (MNIST images) into discrete symbols (AIM sequences) to enable
# endogenous communication in multi-agent reinforcement learning (MARL), facilitating semantic compression (Page 16).
# =======================
class Encoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        # Defines the encoder network to map input images (x_i) to a continuous latent representation z_i (Page 6, Eq. z_i = Enc_A(x_i)).
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, D)
        )

    def forward(self, x):
        # Encodes input image x into a continuous latent vector z_i of dimension D (Page 6).
        return self.enc(x)

class Decoder(nn.Module):
    def __init__(self, D):
        super().__init__()
        # Defines the decoder to reconstruct the input from quantized latent representations z_k (Page 7, Eq. \hat{z} = Decoder(z_k)).
        self.dec = nn.Sequential(
            nn.Linear(D, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
        )

    def forward(self, z_q):
        # Reconstructs the input image from the quantized latent vector z_q, reshaping to match MNIST dimensions (Page 7).
        return self.dec(z_q).view(-1, 1, 28, 28)

class VectorQuantizer(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.K = K
        self.D = D
        # Initializes the codebook with K embeddings of dimension D, uniformly distributed (Page 7, Section 3.1.1).
        self.codebook = nn.Embedding(K, D)
        self.codebook.weight.data.uniform_(-1/K, 1/K)

    def forward(self, z_e):
        # Quantizes continuous latent vectors z_e to discrete symbols z_k by finding the nearest codebook vector (Page 7, Eq. k* = arg min ||z_k - z_e||_2^2).
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
        # Full VQ-VAE forward pass: encodes input x to z_e, quantizes to z_q, and reconstructs to x_hat (Page 6-7).
        # Returns reconstructed image, continuous latent, quantized latent, and codebook indices for downstream use.
        z_e = self.encoder(x)
        z_q, encoding_inds = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, z_e, z_q, encoding_inds

# =======================
# Agent A: Active Communicator with Centralized Critic
# Implements Agent A as described in the paper (Page 11, Section 3), using an Actor-Critic architecture.
# Agent A generates AIM sequences and evaluates joint value using a centralized Critic, incorporating opponent’s AIM sequences (Page 6).
# =======================
class AgentA(nn.Module):
    def __init__(self, vqvae, aim_seq_len=2, K=16):
        super().__init__()
        self.vqvae = vqvae
        self.aim_seq_len = aim_seq_len
        self.K = K
        # Embedding for MNIST labels to incorporate contextual information (Page 11).
        self.label_embed = nn.Embedding(10, 8)
        policy_input_dim = vqvae.encoder.enc[-1].out_features + self.label_embed.embedding_dim

        # Actor (Policy Network): Generates AIM sequence logits based on image encoding and label (Page 11).
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, aim_seq_len * self.K)
        )
        
        # Critic (Value Network): Estimates joint reward using Agent A’s input and Agent B’s AIM sequence (Page 6, centralized Critic).
        critic_input_dim = policy_input_dim + aim_seq_len * self.vqvae.quantizer.D 
        self.value_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256), 
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Opponent AIM Predictor: Predicts Agent B’s AIM sequence to model opponent behavior (Page 15, reflection strategy).
        self.opponent_aim_predictor = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, aim_seq_len * self.K)
        )
        self.aim_embedding = nn.Embedding(self.K, self.vqvae.quantizer.D)

        # Intent Predictor: Maps Agent A’s AIM sequence to cooperation/defection intent (Page 15, Method 1: contextual meaning).
        self.intent_predictor_A = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, label, opponent_aim_sequence=None, mode='policy', self_aim_for_prediction=None, own_aim_for_intent=None):
        # Encodes input image to continuous latent representation z_e (Page 6).
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)
        combined_base_input = torch.cat([z_e, label_feat], dim=1)

        if mode == 'policy':
            # Policy mode: Generates AIM sequence logits and estimates joint value (Page 8, policy gradient).
            aim_logits = self.policy_net(combined_base_input)
            if opponent_aim_sequence is None:
                pass 
            embedded_opponent_aim = self.aim_embedding(opponent_aim_sequence)
            flattened_opponent_aim = embedded_opponent_aim.flatten(start_dim=1)
            combined_critic_input = torch.cat([combined_base_input, flattened_opponent_aim], dim=1)
            value = self.value_net(combined_critic_input)
            return aim_logits.view(-1, self.aim_seq_len, self.K), value.squeeze(-1)
            
        elif mode == 'predict_opponent_aim':
            # Predicts Agent B’s AIM sequence based on Agent A’s AIM and label (Page 15, Method 2: opponent prediction).
            if self_aim_for_prediction is None:
                raise ValueError("self_aim_for_prediction must be provided for 'predict_opponent_aim' mode.")
            embedded_self_aim = self.aim_embedding(self_aim_for_prediction)
            flattened_self_aim = embedded_self_aim.flatten(start_dim=1)
            combined_predictor_input = torch.cat([flattened_self_aim, label_feat], dim=1)
            return self.opponent_aim_predictor(combined_predictor_input).view(-1, self.aim_seq_len, self.K)
        
        elif mode == 'predict_own_intent':
            # Predicts Agent A’s intent (C/D) from its own AIM sequence (Page 15, Method 1).
            if own_aim_for_intent is None:
                raise ValueError("own_aim_for_intent must be provided for 'predict_own_intent' mode.")
            embedded_own_aim = self.aim_embedding(own_aim_for_intent)
            flattened_own_aim = embedded_own_aim.flatten(start_dim=1)
            combined_input_for_intent = torch.cat([flattened_own_aim, label_feat], dim=1)
            return self.intent_predictor_A(combined_input_for_intent)
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentA.")

# =======================
# Agent B: Responsive Communicator
# Implements Agent B, which responds to Agent A’s AIM sequence, incorporating raw image encoding (Page 13).
# Uses a policy network to generate its own AIM sequence and predict opponent intent (Page 15).
# =======================
class AgentB(nn.Module):
    def __init__(self, vqvae, aim_seq_len=2, K=16):
        super().__init__()
        self.vqvae = vqvae
        self.aim_seq_len = aim_seq_len
        self.K = K
        self.embedding = nn.Embedding(self.K, self.vqvae.quantizer.D)
        self.label_embed = nn.Embedding(10, 8)
        # Policy input includes Agent A’s AIM sequence, label, and raw image encoding z_e (Page 13).
        policy_input_dim = (aim_seq_len * self.vqvae.quantizer.D + 
                            self.label_embed.embedding_dim + 
                            vqvae.encoder.enc[-1].out_features)
        self.policy_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, aim_seq_len * self.K)
        )
        # Predicts Agent A’s AIM sequence (Page 15, reflection strategy).
        self.opponent_aim_predictor = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim + vqvae.encoder.enc[-1].out_features, 64),
            nn.ReLU(),
            nn.Linear(64, aim_seq_len * self.K)
        )
        # Decodes Agent A’s intent from received AIM sequence (Page 15, Method 1).
        self.intent_decoder_B = nn.Sequential(
            nn.Linear(aim_seq_len * self.vqvae.quantizer.D + self.label_embed.embedding_dim + vqvae.encoder.enc[-1].out_features, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, received_aim_sequence, label, x, mode='policy', actual_response_aim=None):
        embedded_aim = self.embedding(received_aim_sequence)
        label_feat = self.label_embed(label)
        z_e_from_x = self.vqvae.encoder(x)
        flattened_embedded_aim = embedded_aim.flatten(start_dim=1)
        combined_input_for_nets = torch.cat([flattened_embedded_aim, label_feat, z_e_from_x], dim=1)

        if mode == 'policy':
            # Generates Agent B’s AIM sequence logits based on received AIM, label, and image encoding (Page 13).
            aim_logits = self.policy_net(combined_input_for_nets)
            return aim_logits.view(-1, self.aim_seq_len, self.K)
            
        elif mode == 'predict_opponent_aim':
            # Predicts Agent A’s AIM sequence (Page 15, Method 2).
            return self.opponent_aim_predictor(combined_input_for_nets).view(-1, self.aim_seq_len, self.K)
        
        elif mode == 'decode_opponent_intent':
            # Decodes Agent A’s intent from received AIM sequence (Page 15, Method 1).
            return self.intent_decoder_B(combined_input_for_nets)
        
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentB.")

# =======================
# Game Logic & RL Components
# Defines the game logic and reinforcement learning framework, including payoff and training functions.
# Implements a variant of the Prisoner's Dilemma with contextual rewards (Page 16) and REINFORCE for policy learning (Page 14).
# =======================
def interpret_aim_as_action(aim_sequence_tensor, K):
    # Interprets AIM sequences as human-interpretable actions (C=Cooperate, D=Defect) based on the first symbol (Page 16).
    if aim_sequence_tensor[0] < K // 2:
        return 'C'
    else:
        return 'D'

def classic_pd_payoff(action_A, action_B):
    # Implements the classic Prisoner's Dilemma payoff matrix (Page 16).
    if action_A == 'C' and action_B == 'C': return 3, 3
    if action_A == 'C' and action_B == 'D': return -1, 5
    if action_A == 'D' and action_B == 'C': return 5, -1
    if action_A == 'D' and action_B == 'D': return 0, 0

def payoff(action_A, action_B, image_label, current_round):
    # Defines a contextual payoff function influenced by the MNIST label, encouraging cooperation for even labels (Page 16).
    rA, rB = 0, 0
    if action_A == 'C' and action_B == 'C': 
        rA, rB = 4, 4 
    elif action_A == 'C' and action_B == 'D': 
        rA, rB = -1, 5 
    elif action_A == 'D' and action_B == 'C': 
        rA, rB = 5, -1
    elif action_A == 'D' and action_B == 'D': 
        rA, rB = 0, 0 
    if image_label % 2 == 0:
        if action_A == 'C' and action_B == 'C':
            rA += 1 
            rB += 1
    else:
        if action_A == 'C' and action_B == 'D':
            rA -= 1 
        elif action_A == 'D' and action_B == 'C':
            rB -= 1 
    return rA, rB

def train_vqvae(epochs, K_val, D_val):
    # Trains the VQ-VAE to learn discrete representations of MNIST images (Page 16, Section 5.2.1).
    # Uses reconstruction loss, commitment loss, and codebook loss as described in the paper (Page 14).
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

def multi_agent_game(vqvae, aim_dict, rounds=5, aim_seq_len=2, K_val=16, 
                     reflection_strategy='none', reflection_coeff=0.1, gamma_rl=0.99, entropy_coeff=0.01): 
    # Implements the multi-agent game using the AIM framework, with REINFORCE and reflection strategies (Pages 14-15).
    # Freezes VQ-VAE parameters to focus on policy learning (Page 7).
    for param in vqvae.parameters():
        param.requires_grad = False
    agentA = AgentA(vqvae, aim_seq_len, K_val)
    agentB = AgentB(vqvae, aim_seq_len, K_val)
    optimizer_A = optim.Adam(list(agentA.parameters()), lr=1e-4)
    optimizer_B = optim.Adam(list(agentB.parameters()), lr=1e-4)
    scheduler_A = torch.optim.lr_scheduler.ExponentialLR(optimizer_A, gamma=0.9995)
    scheduler_B = torch.optim.lr_scheduler.ExponentialLR(optimizer_B, gamma=0.9995)
    transform = transforms.ToTensor()
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    A_rewards_history, B_rewards_history = [], []
    all_labels = torch.arange(10).repeat(rounds // 10 + 1)[:rounds].tolist()
    random.shuffle(all_labels)
    label_to_indices = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(test_data):
        label_to_indices[label].append(idx)
    reflection_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    intent_loss_fn = nn.CrossEntropyLoss()
    initial_entropy_coeff = entropy_coeff
    entropy_decay_rate = 0.9999

    print(f"\n--- Starting Multi-Agent AIM Game (Strategy: {reflection_strategy}, Enhanced Complexity, Centralized Critic) ---")
    for i in range(rounds):
        current_label = all_labels[i]
        matching_indices = label_to_indices[current_label]
        if not matching_indices:
            idx = random.randint(0, len(test_data)-1)
            x, _ = test_data[idx]
        else:
            idx = random.choice(matching_indices)
            x, _ = test_data[idx]
        x = x.unsqueeze(0)
        current_label_tensor = torch.tensor([current_label])

        # Agent A generates AIM sequence using policy network (Page 11).
        A_aim_logits_policy, _ = agentA(x, current_label_tensor, mode='policy', opponent_aim_sequence=torch.zeros((1,aim_seq_len), dtype=torch.long))
        A_dist = torch.distributions.Categorical(logits=A_aim_logits_policy.view(-1, K_val))
        A_sampled_aim_flat = A_dist.sample()
        A_sampled_aim_sequence = A_sampled_aim_flat.view(1, -1)
        A_log_probs = A_dist.log_prob(A_sampled_aim_flat).sum()
        A_entropy = A_dist.entropy().sum()

        # Agent B responds with its AIM sequence, incorporating raw image encoding (Page 13).
        B_aim_logits_policy = agentB(A_sampled_aim_sequence, current_label_tensor, x, mode='policy')
        B_dist = torch.distributions.Categorical(logits=B_aim_logits_policy.view(-1, K_val))
        B_sampled_aim_flat = B_dist.sample()
        B_sampled_aim_sequence = B_sampled_aim_flat.view(1, -1)
        B_log_probs = B_dist.log_prob(B_sampled_aim_flat).sum()
        B_entropy = B_dist.entropy().sum()

        # Interpret AIM sequences as actions (C/D) for reward calculation (Page 16).
        A_action_human_interp = interpret_aim_as_action(A_sampled_aim_sequence[0], K_val)
        B_action_human_interp = interpret_aim_as_action(B_sampled_aim_sequence[0], K_val)

        # Compute contextual rewards based on actions and MNIST label (Page 16).
        A_reward_indiv, B_reward_indiv = payoff(
            A_action_human_interp, B_action_human_interp, current_label, i + 1 
        )
        joint_reward = A_reward_indiv + B_reward_indiv 

        # Centralized Critic evaluation by Agent A (Page 6).
        A_critic_input_z_e = agentA.vqvae.encoder(x)
        A_critic_input_label_feat = agentA.label_embed(current_label_tensor)
        A_critic_input_combined_policy = torch.cat([A_critic_input_z_e, A_critic_input_label_feat], dim=1)
        B_aim_embedded = agentA.aim_embedding(B_sampled_aim_sequence)
        B_aim_flattened = B_aim_embedded.flatten(start_dim=1)
        A_critic_total_input = torch.cat([A_critic_input_combined_policy, B_aim_flattened], dim=1)
        A_value = agentA.value_net(A_critic_total_input).squeeze(-1)

        # Compute A2C loss using joint reward and advantage (Page 8, policy gradient).
        A_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_A_policy = - (A_log_probs * A_advantage.to(A_value.device))
        loss_A_value = value_loss_fn(A_value, torch.tensor([joint_reward], dtype=torch.float32).to(A_value.device))
        current_entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** i)
        loss_A = loss_A_policy + 0.5 * loss_A_value - current_entropy_coeff * A_entropy

        # Agent B policy loss using shared advantage from Agent A’s Critic (Page 8).
        B_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_B_policy = - (B_log_probs * B_advantage.to(A_value.device))
        loss_B = loss_B_policy - current_entropy_coeff * B_entropy

        # Intent alignment loss for Agent A (Page 15, Method 1).
        target_A_action_idx = torch.tensor([0 if A_action_human_interp == 'C' else 1], dtype=torch.long).to(A_value.device)
        target_B_action_idx = torch.tensor([0 if B_action_human_interp == 'C' else 1], dtype=torch.long).to(A_value.device)
        predicted_A_intent_logits = agentA(x, current_label_tensor, 
                                           mode='predict_own_intent', 
                                           own_aim_for_intent=A_sampled_aim_sequence)
        loss_A_own_intent = intent_loss_fn(predicted_A_intent_logits, target_A_action_idx)
        loss_A += reflection_coeff * loss_A_own_intent

        # Intent alignment loss for Agent B (Page 15, Method 1).
        predicted_B_decoded_intent_logits = agentB(A_sampled_aim_sequence, current_label_tensor, x,
                                                   mode='decode_opponent_intent')
        loss_B_decode_intent = intent_loss_fn(predicted_B_decoded_intent_logits, target_A_action_idx)
        loss_B += reflection_coeff * loss_B_decode_intent

        # Predictive bias reflection strategy (Page 15, Method 2).
        if reflection_strategy == 'predictive_bias':
            predicted_B_aim_logits_by_A = agentA(x, current_label_tensor, 
                                                mode='predict_opponent_aim', 
                                                self_aim_for_prediction=A_sampled_aim_sequence)
            loss_A_predictive_bias = reflection_loss_fn(
                predicted_B_aim_logits_by_A.permute(0, 2, 1), 
                B_sampled_aim_sequence.long()
            )
            loss_A += reflection_coeff * loss_A_predictive_bias
            predicted_A_aim_logits_by_B = agentB(A_sampled_aim_sequence, current_label_tensor, x,
                                                mode='predict_opponent_aim')
            loss_B_predictive_bias = reflection_loss_fn(
                predicted_A_aim_logits_by_B.permute(0, 2, 1), 
                A_sampled_aim_sequence.long()
            )
            loss_B += reflection_coeff * loss_B_predictive_bias
        elif reflection_strategy == 'aim_context_value':
            print("Warning: 'aim_context_value' strategy is not compatible with current agent's 'aim_eval_net' design.")
            pass

        # Backpropagate and update agent policies (Page 8).
        loss_A.backward()
        optimizer_A.step()
        loss_B.backward()
        optimizer_B.step()
        scheduler_A.step()
        scheduler_B.step()

        # Log AIM sequences and their interpretations in the AIM dictionary (Page 21).
        context = f"PD Round {i+1} (Label: {current_label})"
        aim_dict.add_entry(str(A_sampled_aim_sequence.tolist()[0]), A_action_human_interp, context)
        aim_dict.add_entry(str(B_sampled_aim_sequence.tolist()[0]), B_action_human_interp, context + " (Response)")
        A_rewards_history.append(A_reward_indiv)
        B_rewards_history.append(B_reward_indiv)
        
        # Print round results, including AIM sequences and rewards (Page 21).
        print(f'Round {i+1}/{rounds}: '
              f'Label={current_label} | '
              f'Agent A AIM Seq={A_sampled_aim_sequence.tolist()[0]} (Interp: {A_action_human_interp}) | '
              f'Agent B AIM Seq={B_sampled_aim_sequence.tolist()[0]} (Interp: {B_action_human_interp}) | '
              f'Reward A={A_reward_indiv:.2f}, B={B_reward_indiv:.2f} | Avg A={sum(A_rewards_history) / len(A_rewards_history):.2f}, Avg B={sum(B_rewards_history) / len(B_rewards_history):.2f}')

    return A_rewards_history, B_rewards_history

def visualize(A_rewards, B_rewards, strategy_name):
    # Visualizes total rewards over rounds, as shown in the paper’s performance plots (Page 20, Figure 3).
    total_rewards = [a + b for a, b in zip(A_rewards, B_rewards)]
    plt.figure(figsize=(10, 6))
    plt.plot(total_rewards, label='Total Reward (A + B)', alpha=0.7, color='purple')
    plt.title(f'Total Payoff Over Time (Strategy: {strategy_name})')
    plt.xlabel('Round')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# =======================
# Main Execution with CLI
# Sets up the experiment with configurable parameters, aligning with the paper’s experimental setup (Page 16).
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Agent AIM Game with Reflection Strategies and Enhanced Complexity (Centralized Critic)")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for VQ-VAE')
    parser.add_argument('--rounds', type=int, default=10000, help='Number of multi-agent game rounds (more for RL)')
    parser.add_argument('--K', type=int, default=32, help='Size of the VQ-VAE codebook (number of AIM symbols)')
    parser.add_argument('--D', type=int, default=64, help='Dimension of the VQ-VAE code vectors')
    parser.add_argument('--aim_seq_len', type=int, default=2, help='Length of the AIM symbol sequence for communication')
    parser.add_argument('--reflection_strategy', type=str, default='predictive_bias', 
                        choices=['none', 'aim_context_value', 'predictive_bias'],
                        help='Reflection strategy to use: none, aim_context_value, or predictive_bias')
    parser.add_argument('--reflection_coeff', type=float, default=0.05, 
                        help='Coefficient for the reflection loss term')
    parser.add_argument('--gamma_rl', type=float, default=0.99, help='Discount factor for RL rewards (gamma_rl in A2C)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Initial coefficient for entropy regularization')
    args = parser.parse_args()

    aim_dict = AIMDictionary()
    vqvae = train_vqvae(args.epochs, args.K, args.D)
    A_rewards, B_rewards = multi_agent_game(vqvae, aim_dict, rounds=args.rounds,
                                            aim_seq_len=args.aim_seq_len, K_val=args.K,
                                            reflection_strategy=args.reflection_strategy,
                                            reflection_coeff=args.reflection_coeff,
                                            gamma_rl=args.gamma_rl,
                                            entropy_coeff=args.entropy_coeff)
    aim_dict.save()
    visualize(A_rewards, B_rewards, args.reflection_strategy)