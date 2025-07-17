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
import math

# =======================
# AIMDictionary
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
# VQ-VAE
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
# Agents (Modified to update AgentB's forward function)
# =======================
class AgentA(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.label_embed = nn.Embedding(10, 8)
        self.action_embed = nn.Embedding(2, 8)

        policy_input_dim = vqvae.encoder.enc[-1].out_features + self.label_embed.embedding_dim

        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        critic_input_dim = policy_input_dim + self.action_embed.embedding_dim
        self.value_net = nn.Sequential(
            nn.Linear(critic_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.intent_predictor_A = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, label, opponent_action=None, mode='policy'):
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)
        combined_base_input = torch.cat([z_e, label_feat], dim=1)

        if mode == 'policy':
            action_logits = self.policy_net(combined_base_input)

            if opponent_action is None:
                dummy_action = torch.tensor([0]).to(label.device)
                embedded_opponent_action = self.action_embed(dummy_action)
            else:
                embedded_opponent_action = self.action_embed(opponent_action)

            combined_critic_input = torch.cat([combined_base_input, embedded_opponent_action], dim=1)
            value = self.value_net(combined_critic_input)

            return action_logits, value.squeeze(-1)

        elif mode == 'predict_own_intent':
            return self.intent_predictor_A(combined_base_input)

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentA.")

class AgentB(nn.Module):
    def __init__(self, vqvae):
        super().__init__()
        self.vqvae = vqvae
        self.label_embed = nn.Embedding(10, 8)
        self.action_embed = nn.Embedding(2, 8) # 0 for C, 1 for D

        policy_input_dim = (self.vqvae.encoder.enc[-1].out_features +
                            self.label_embed.embedding_dim +
                            self.action_embed.embedding_dim)

        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.intent_decoder_B = nn.Sequential(
            nn.Linear(policy_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    # Forward function: Supports 'policy_no_message' mode
    def forward(self, x, label, opponent_action=None, mode='policy'):
        z_e = self.vqvae.encoder(x)
        label_feat = self.label_embed(label)

        # Handle case where opponent_action is None (for positive listening with no message)
        if opponent_action is None:
            # Create a "no message" embedding. Assumes action_embed's embedding_dim is 8.
            # A zero vector is a common way to represent "no message received."
            embedded_opponent_action = torch.zeros(label.shape[0], self.action_embed.embedding_dim, device=x.device)
        else:
            embedded_opponent_action = self.action_embed(opponent_action)

        combined_input_for_nets = torch.cat([z_e, label_feat, embedded_opponent_action], dim=1)

        if mode == 'policy':
            action_logits = self.policy_net(combined_input_for_nets)
            return action_logits

        elif mode == 'decode_opponent_intent':
            return self.intent_decoder_B(combined_input_for_nets)

        elif mode == 'policy_no_message': # Mode for computing policy without a message
            # In this mode, opponent_action should be None, using a zero embedding
            # Redefine no-message input for clarity
            embedded_no_message = torch.zeros(label.shape[0], self.action_embed.embedding_dim, device=x.device)
            combined_input_no_message = torch.cat([z_e, label_feat, embedded_no_message], dim=1)
            action_logits_no_message = self.policy_net(combined_input_no_message)
            return action_logits_no_message

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented for AgentB.")

# =======================
# Game Logic & RL Components (Modified multi_agent_game function)
# =======================

def payoff(action_A, action_B, image_label, current_round):
    rA, rB = 0, 0

    # A. Base rewards
    if action_A == 'C' and action_B == 'C':
        rA, rB = 3, 3
    elif action_A == 'C' and action_B == 'D':
        rA, rB = -1, 5
    elif action_A == 'D' and action_B == 'C':
        rA, rB = 5, -1
    elif action_A == 'D' and action_B == 'D':
        rA, rB = 0, 0

    # B. Contextual rewards
    if image_label % 2 == 0:  # Even labels: Additional incentive for cooperation
        if action_A == 'C' and action_B == 'C':
            rA += 2 # From (3,3) to (5,5)
            rB += 2
        elif (action_A == 'C' and action_B == 'D'):
            rA -= 1 # Penalty for cooperating
        elif (action_A == 'D' and action_B == 'C'):
            rB -= 1 # Penalty for cooperating
    else:  # Odd labels: Slight penalty for unilateral cooperation, but C,C remains optimal
        # Reward calculation is the same as for even labels, but C,C has no additional reward
        # Thus, C,C remains (3,3)
        if (action_A == 'C' and action_B == 'D'):
            rA -= 1 # Penalty for cooperating
        elif (action_A == 'D' and action_B == 'C'):
            rB -= 1 # Penalty for cooperating

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

# Modified multi_agent_game function to include new hyperparameters
def multi_agent_game(vqvae, aim_dict, rounds=5,
                     reflection_strategy='intent_alignment', reflection_coeff=0.1, gamma_rl=0.99, entropy_coeff=0.01,
                     ps_coeff=0.1, ps_target_entropy=None, pl_coeff=0.1):

    # Freeze VQ-VAE parameters
    for param in vqvae.parameters():
        param.requires_grad = False

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

    # For Positive Signalling loss: Accumulate AgentA's policy logits
    # Simplified approach; a more robust implementation might use a replay buffer or EMA
    batch_A_policy_logits_for_ps = []

    print(f"\n--- Starting Multi-Agent Contextual Prisoner's Dilemma Game (Strategy: {reflection_strategy}) ---")
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

        # 1. Agent action generation (Actor part)
        A_action_logits_policy, _ = agentA(x, current_label_tensor, mode='policy', opponent_action=torch.tensor([0]))
        A_dist = torch.distributions.Categorical(logits=A_action_logits_policy)
        A_sampled_action = A_dist.sample() # 0 (C) or 1 (D)
        A_log_probs = A_dist.log_prob(A_sampled_action)
        A_entropy = A_dist.entropy() # This is H(m_t^i|x_t^i)

        # AgentB policy output, receiving AgentA's action (this is AgentB's policy with message)
        B_action_logits_policy = agentB(x, current_label_tensor, A_sampled_action, mode='policy')
        B_dist = torch.distributions.Categorical(logits=B_action_logits_policy)
        B_sampled_action = B_dist.sample() # 0 (C) or 1 (D)
        B_log_probs = B_dist.log_prob(B_sampled_action)
        B_entropy = B_dist.entropy()

        # Convert numerical actions to C/D strings
        A_action_human_interp = 'C' if A_sampled_action.item() == 0 else 'D'
        B_action_human_interp = 'C' if B_sampled_action.item() == 0 else 'D'

        # 2. Compute rewards
        A_reward_indiv, B_reward_indiv = payoff(
            A_action_human_interp, B_action_human_interp, current_label, i + 1
        )
        joint_reward = A_reward_indiv + B_reward_indiv

        # 3. Compute centralized Critic's value (in AgentA)
        _, A_value = agentA(x, current_label_tensor, mode='policy', opponent_action=B_sampled_action)

        # 4. Compute core A2C loss
        # Agent A loss (Actor-Critic)
        A_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_A_policy = - (A_log_probs * A_advantage.to(A_value.device))
        loss_A_value = value_loss_fn(A_value, torch.tensor([joint_reward], dtype=torch.float32).to(A_value.device))

        current_entropy_coeff = initial_entropy_coeff * (entropy_decay_rate ** i)
        loss_A = loss_A_policy + 0.5 * loss_A_value - current_entropy_coeff * A_entropy

        # Agent B loss (policy loss based on shared Critic's Advantage)
        B_advantage = torch.tensor([joint_reward], dtype=torch.float32) - A_value.cpu().detach()
        loss_B_policy = - (B_log_probs * B_advantage.to(A_value.device))

        loss_B = loss_B_policy - current_entropy_coeff * B_entropy

        # Positive Signalling loss
        if ps_coeff > 0:
            current_cond_entropy = A_entropy # H(m_t^i|x_t^i)
            batch_A_policy_logits_for_ps.append(A_action_logits_policy.detach())

            # Ensure enough data in batch to estimate average policy
            # Simplified approach; a more accurate implementation might use a replay buffer or EMA
            if len(batch_A_policy_logits_for_ps) >= 1: # At least one sample, but more are preferred
                # Average logits in the batch
                avg_A_logits = torch.stack(batch_A_policy_logits_for_ps).mean(dim=0)
                avg_A_prob = torch.softmax(avg_A_logits, dim=-1)
                H_avg_A_msg = -(avg_A_prob * torch.log(avg_A_prob + 1e-9)).sum() # H(overline{pi_M^i})

                if ps_target_entropy is None:
                    # For 2 discrete actions (C/D), maximum entropy is log(2)
                    # Paper suggests target_entropy as log(|A|)/2
                    target_H_val = math.log(2.0) / 2
                else:
                    target_H_val = ps_target_entropy

                # L_ps = - (lambda * H(overline{pi_M^i}) - (H(m_t^i|x_t) - H_target)^2)
                # Lambda is positive, and the loss is minimized, so the negative sign is correct
                loss_ps = -(H_avg_A_msg - (current_cond_entropy - target_H_val)**2)
                loss_A += ps_coeff * loss_ps

            # Optional: Periodically clear or limit batch size to control computational cost and bias
            if len(batch_A_policy_logits_for_ps) > 64: # Clear every 64 samples
                batch_A_policy_logits_for_ps = []

        # Positive Listening loss
        if pl_coeff > 0:
            # pi_A^i(a|x_t) is AgentB's action probability with message
            B_prob_with_message = torch.softmax(B_action_logits_policy, dim=-1)

            # overline{pi_A^i}(a|x_t') is AgentB's action probability without message
            # Ensure this computation does not affect gradients, as it's a baseline
            with torch.no_grad():
                B_action_logits_no_message = agentB(x, current_label_tensor, opponent_action=None, mode='policy_no_message')
                B_prob_no_message = torch.softmax(B_action_logits_no_message, dim=-1)

            # L_pl = - sum | pi_A^i(a|x_t) - overline{pi_A^i}(a|x_t') |
            # Maximize L1 norm, so loss is negative L1 norm
            loss_pl = -torch.sum(torch.abs(B_prob_with_message - B_prob_no_message), dim=-1).mean()
            loss_B += pl_coeff * loss_pl

        # 5. Intent alignment loss
        if reflection_strategy == 'intent_alignment':
            target_A_action_idx = A_sampled_action # 0 or 1

            # Agent A predicts its own intent (input is context, output is its own action prediction)
            predicted_A_intent_logits = agentA(x, current_label_tensor, mode='predict_own_intent')
            loss_A_own_intent = intent_loss_fn(predicted_A_intent_logits, target_A_action_idx)
            loss_A += reflection_coeff * loss_A_own_intent

            # Agent B decodes opponent's intent (input is context and opponent's action, output is opponent's action prediction)
            predicted_B_decoded_intent_logits = agentB(x, current_label_tensor, A_sampled_action, mode='decode_opponent_intent')
            loss_B_decode_intent = intent_loss_fn(predicted_B_decoded_intent_logits, target_A_action_idx)
            loss_B += reflection_coeff * loss_B_decode_intent

        # Clear gradients and perform backpropagation and optimization
        optimizer_A.zero_grad()
        loss_A.backward()
        optimizer_A.step()

        optimizer_B.zero_grad()
        loss_B.backward()
        optimizer_B.step()

        scheduler_A.step()
        scheduler_B.step()

        # Record rewards
        A_rewards_history.append(A_reward_indiv)
        B_rewards_history.append(B_reward_indiv)
        Joint_rewards_history.append(joint_reward)

        # Log game data to AIMDictionary
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
    # Plot only joint rewards
    plt.plot(Joint_rewards, label='Joint Reward', alpha=0.7, linestyle='--', color='red')
    plt.title(f'Payoff Over Time (Strategy: {strategy_name}) with RIAL - Joint Reward Only')
    plt.xlabel('Round')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

# =======================
# Main with CLI (Updated argument parsing)
# =======================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Agent Contextual Prisoner's Dilemma Game with Intent Alignment and Emergent Communication Biases")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs for VQ-VAE')
    parser.add_argument('--rounds', type=int, default=10000, help='Number of multi-agent game rounds (more for RL)')
    parser.add_argument('--K', type=int, default=32, help='Size of the VQ-VAE codebook')
    parser.add_argument('--D', type=int, default=64, help='Dimension of the VQ-VAE code vectors')
    parser.add_argument('--reflection_strategy', type=str, default='intent_alignment',
                        choices=['none', 'intent_alignment'],
                        help='Reflection strategy to use: none or intent_alignment (default)')
    parser.add_argument('--reflection_coeff', type=float, default=0.05,
                        help='Coefficient for the reflection loss term')
    parser.add_argument('--gamma_rl', type=float, default=0.99, help='Discount factor for RL rewards (gamma_rl in A2C)')
    parser.add_argument('--entropy_coeff', type=float, default=0.01, help='Initial coefficient for entropy regularization')

    # New parameters for Positive Signalling and Positive Listening
    parser.add_argument('--ps_coeff', type=float, default=0.0, # Disabled by default
                        help='Coefficient for Positive Signalling loss. Set > 0 to enable')
    parser.add_argument('--ps_target_entropy', type=float, default=None,
                        help='Target conditional entropy for Positive Signalling. If None, defaults to log(2)/2 for 2 actions')
    parser.add_argument('--pl_coeff', type=float, default=0.0, # Disabled by default
                        help='Coefficient for Positive Listening loss. Set > 0 to enable')

    args = parser.parse_args()

    aim_dict = AIMDictionary()
    vqvae = train_vqvae(args.epochs, args.K, args.D)

    A_rewards, B_rewards, Joint_rewards = multi_agent_game(vqvae, aim_dict, rounds=args.rounds,
                                            reflection_strategy=args.reflection_strategy,
                                            reflection_coeff=args.reflection_coeff,
                                            gamma_rl=args.gamma_rl,
                                            entropy_coeff=args.entropy_coeff,
                                            ps_coeff=args.ps_coeff,
                                            ps_target_entropy=args.ps_target_entropy,
                                            pl_coeff=args.pl_coeff)

    aim_dict.save()
    visualize(A_rewards, B_rewards, Joint_rewards, args.reflection_strategy)