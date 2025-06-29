# Multi-Agent Collaborative Game with Inductive Biases: Intent Alignment and Emergent Communication

-----

## Project Overview

This project is one of a series of research outcomes developed based on my proposed **"AI Mother Tongue Framework."** This framework aims to provide a new perspective and approach for communication learning in multi-agent systems.

Currently, in multi-agent environments, especially when facing complex tasks, **academia appears to have yet to achieve significant breakthroughs in effectively promoting communication among agents.** However, the **"AI Mother Tongue Framework" has demonstrated remarkably good results.** My AIM is visualized in `vqvae_agents_game_AIM.jpg`.

For comparative experiments, we benchmarked our findings against **DeepMind's RIAT (Reinforced Intent-Aligned Training) method.** While RIAT showed good performance in the simpler tasks designed in its original paper, it's worth noting that **in the complex tasks designed for this study, even with the introduction of RIAT's inductive biases (Positive Signalling and Positive Listening), its results still showed a failure to converge.**

-----

## Code Files and Experimental Comparisons

A paper on this project is currently being written, so the specific methods of this project are not being disclosed at this time. However, for comparative purposes, this project designed a relatively complex task to observe the behavior of two agents. The currently released code files are all for the control groups, including two main game execution scripts:

  * **`vqvae_agents_game_RIAL.py`**:
    This file implements **DeepMind's RIAT (Reinforced Intent-Aligned Training) method**, specifically the **Positive Signalling** and **Positive Listening** inductive biases mentioned in their paper. The test results are stored in `game_log.json`, and the experimental results are visualized in `vqvae_agents_game_RIAL.jpg`.

  * **`vqvae_agents_game.py`**:
    This file is based on the original "Intent Alignment" framework and **does not include any RIAT inductive biases**. Its test results are stored in `game_log.json.old`, and the experimental results are visualized in `vqvae_agents_game.jpg`.

-----

## Task Design

Agent rewards are based on their taken actions (`action_A`, `action_B`) and the image label (`image_label`).

The reward function is designed with a structure referencing the Prisoner's Dilemma, and it incorporates image labels to modulate the rewards, significantly increasing task complexity. The architecture diagram is shown in `vqvae_agents_game_mission.jpg`.

-----

## Experimental Control Group Code Architecture Analysis (Project's specific methods are not disclosed at this time; the following only pertains to the RIAT inductive biases of the control group)

In multi-agent systems, learning to communicate often presents challenges. This research introduces and tests two key inductive biases that actively encourage agents to develop communication behaviors through the design of specific loss functions:

### 1\. Positive Signalling (AgentA Related)

  * **Goal**: Encourage the sender (AgentA) to transmit different messages (i.e., C/D actions) in varying situations, making their messages informative rather than merely random. This ensures that the message carries contextual information.

### 2\. Positive Listening (AgentB Related)

  * **Goal**: Encourage the listener (AgentB) to adjust its action strategy based on the received message (AgentA's action). This ensures that the listener "pays attention" and "responds" to the communication channel.

-----

## Code Implementation Details

This project extends the original "Intent Alignment" multi-agent cooperative game code by primarily adding the implementation of **Positive Signalling loss ($L\_{ps}$)** and **Positive Listening loss ($L\_{pl}$)**.

### 1\. Positive Signalling Loss (`loss_ps`)

This loss function primarily acts on the sender, AgentA.

  * **Goal**: Ensure that AgentA's C/D actions are "meaningful," meaning different contexts (defined by MNIST digit label `current_label` and image encoding `z_e`) lead AgentA to take different actions. In other words, AgentA's actions should contain information about its context.

  * **Implementation Details**:

    ```python
    # AgentA policy output
    A_action_logits_policy, _ = agentA(x,
    current_label_tensor, mode='policy', opponent_action=torch.tensor([0]))
    A_dist =
    torch.distributions.Categorical(logits=A_action_logits_policy)
    A_sampled_action = A_dist.sample() # 0 (C) or 1 (D)
    A_log_probs = A_dist.log_prob(A_sampled_action)
    A_entropy = A_dist.entropy() # This is H(m_t^i|x_t^i)

    # ... (other code) ...

    # --- Positive Signalling Loss ---
    if ps_coeff > 0:
        current_cond_entropy = A_entropy # H(m_t^i|x_t^i)
        # Accumulate AgentA's policy logits for estimating average policy
        batch_A_policy_logits_for_ps.append(A_action_logits_policy.detach())

        if len(batch_A_policy_logits_for_ps) >= 1: # At least one sample
            # Calculate the entropy of the average message policy H(overline{pi_M^i})
            avg_A_logits = torch.stack(batch_A_policy_logits_for_ps).mean(dim=0)
            avg_A_prob = torch.softmax(avg_A_logits, dim=-1)
            H_avg_A_msg = -(avg_A_prob * torch.log(avg_A_prob + 1e-9)).sum()

            # Set target conditional entropy
            if ps_target_entropy is None:
                target_H_val = math.log(2.0) / 2 # For 2 actions, half of max entropy log(2)
            else:
                target_H_val = ps_target_entropy

            # Calculate L_ps loss: - (lambda * H(overline{pi_M^i}) - (H(m_t^i|x_t) - H_target)^2)
            loss_ps = -(H_avg_A_msg - (current_cond_entropy - target_H_val)**2)
            loss_A += ps_coeff * loss_ps

            # Clear or limit batch size
            if len(batch_A_policy_logits_for_ps) > 64:
                batch_A_policy_logits_for_ps = []
    ```

### 2\. Positive Listening Loss (`loss_pl`)

This loss function primarily acts on the listener, AgentB.

  * **Goal**: Encourage AgentB's action strategy to be influenced by the messages sent by AgentA. If AgentB's actions are indistinguishable whether a message is received or not, then the communication channel is not being effectively utilized.

  * **Implementation Details**:

    ```python
    class AgentB(nn.Module):
        # ... (original __init__ function) ...
        def forward(self, x, label, opponent_action=None, mode='policy'):
            z_e = self.vqvae.encoder(x)
            label_feat = self.label_embed(label)

            # Handle cases where opponent_action might be None (for Positive Listening's no-message policy)
            if opponent_action is None:
                # Create a "no-message" embedding. Here we assume action_embed's embedding_dim is 8.
                embedded_opponent_action = torch.zeros(label.shape[0], self.action_embed.embedding_dim, device=x.device)
            else:
                embedded_opponent_action = self.action_embed(opponent_action)

            combined_input_for_nets = torch.cat([z_e, label_feat, embedded_opponent_action], dim=1)

            if mode == 'policy':
                action_logits = self.policy_net(combined_input_for_nets)
                return action_logits

            # ... (other modes, e.g., 'decode_opponent_intent') ...

            elif mode == 'policy_no_message': # New mode: for calculating policy when no message
                # In this mode, opponent_action should be None
                embedded_no_message = torch.zeros(label.shape[0], self.action_embed.embedding_dim, device=x.device)
                combined_input_no_message = torch.cat([z_e, label_feat, embedded_no_message], dim=1)
                action_logits_no_message = self.policy_net(combined_input_no_message)
                return action_logits_no_message

            else:
                raise NotImplementedError(f"Mode '{mode}' not implemented for AgentB.")
    ```

      * Within the `multi_agent_game` loop, the `loss_pl` calculation logic is as follows:

    <!-- end list -->

    ```python
    # AgentB policy output, receiving AgentA's action (this is AgentB's policy with message)
    B_action_logits_policy = agentB(x,
    current_label_tensor, A_sampled_action, mode='policy')

    # ... (other AgentB related calculations) ...

    # --- Positive Listening Loss ---
    if pl_coeff > 0:
        # pi_A^i(a|x_t) is AgentB's action probability with message
        B_prob_with_message = torch.softmax(B_action_logits_policy, dim=-1)

        # overline{pi_A^i}(a|x_t') is AgentB's action probability without message
        # Ensure this calculation does not affect gradients, as it's a baseline
        with torch.no_grad():
            B_action_logits_no_message = agentB(x, current_label_tensor, opponent_action=None, mode='policy_no_message')
            B_prob_no_message = torch.softmax(B_action_logits_no_message, dim=-1)

        # Calculate L1 norm loss for Positive Listening
        # We want to maximize the L1 norm, so the loss is the negative L1 norm
        loss_pl = -torch.sum(torch.abs(B_prob_with_message - B_prob_no_message), dim=-1).mean()
        loss_B += pl_coeff * loss_pl
    ```

-----

## Command Line Argument Control

For easy experimentation and parameter tuning, this project includes the following arguments in `argparse`, allowing users to control the enabling and weighting of these new losses at runtime:

```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Agent Contextual Prisoner's Dilemma Game with Intent Alignment and Emergent Communication Biases")
    # ... (original arguments) ...

    # New Positive Signalling and Positive Listening parameters
    parser.add_argument('--ps_coeff', type=float, default=0.0, # Default to 0, needs manual enabling
                        help='Coefficient for Positive Signalling loss. Set > 0 to enable.')
    parser.add_argument('--ps_target_entropy', type=float, default=None,
                        help='Target conditional entropy for Positive Signalling. If None, defaults to log(2)/2 for 2 actions.')
    parser.add_argument('--pl_coeff', type=float, default=0.0, # Default to 0, needs manual enabling
                        help='Coefficient for Positive Listening loss. Set > 0 to enable.')

    args = parser.parse_args()

    # ... (VQ-VAE training) ...

    # Pass new parameters to the multi_agent_game function
    A_rewards, B_rewards, Joint_rewards = multi_agent_game(vqvae, aim_dict,
                                            rounds=args.rounds,
                                            reflection_strategy=args.reflection_strategy,
                                            reflection_coeff=args.reflection_coeff,
                                            gamma_rl=args.gamma_rl,
                                            entropy_coeff=args.entropy_coeff,
                                            ps_coeff=args.ps_coeff, # Pass new parameter
                                            ps_target_entropy=args.ps_target_entropy, # Pass new parameter
                                            pl_coeff=args.pl_coeff) # Pass new parameter

    # ... (saving logs and visualization) ...
```

### How to Use These Parameters:

```bash
# Run game without RIAT biases
python vqvae_agents_game.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1

# Run game with RIAT biases
python vqvae_agents_game_RIAL.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1 --ps_coeff 0.1 --pl_coeff 0.05
```

-----

## Installation and Execution

### Environment Setup

This project requires Python 3.x and PyTorch. It's recommended to create a virtual environment using `conda` or `venv`.

1.  **Create a virtual environment** (e.g., using conda):

    ```bash
    conda create -n emergent_comm_env python=3.9
    conda activate emergent_comm_env
    ```

2.  **Install dependencies** (ensure your `requirements.txt` file includes all necessary libraries like `torch`, `torchvision`, `numpy`, `matplotlib`, etc.):

    ```bash
    pip install -r requirements.txt
    ```

### Running Examples

Here are some example commands to run this project:

1.  **Run the game without RIAT biases** (data will be saved to `game_log.json.old`):

    ```bash
    python vqvae_agents_game.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1
    ```

2.  **Run the game with RIAT biases** (data will be saved to `game_log.json`):

    ```bash
    python vqvae_agents_game_RIAL.py --rounds 10000 --reflection_strategy intent_alignment --reflection_coeff 0.1 --ps_coeff 0.1 --pl_coeff 0.05
    ```

-----

## Experimental Results and Visualization

(This section is reserved for your experimental results and visualizations. You can upload charts showing reward curves, changes in communication strategies, agent action distributions, etc. This will significantly strengthen the project's persuasiveness, especially when demonstrating your observations regarding RIAT's performance.)

  * [Figure 1: Joint Rewards over Training Rounds - Comparison with/without RIAT biases]
  * [Figure 2: AgentA Communication Action (C/D) Distribution Comparison]
  * [Figure 3: AgentB Action (Left/Right) Distribution Comparison]

-----

## References

Biases for emergent communication in multi-agent reinforcement learning
https://dl.acm.org/doi/10.5555/3454287.3455463

-----

## License

This project is open-sourced under the [MIT License](https://www.google.com/search?q=LICENSE).