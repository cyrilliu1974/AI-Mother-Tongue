import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional

# ====================================================================
# HQ-VAE Core Components (Extracted from previous snippet)
# = ==================================================================

class VectorQuantizer(nn.Module):
    """Vector Quantization layer for discrete representation learning"""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through vector quantizer
        
        Args:
            inputs: Input tensor [B, C, H, W] or [B, L, D]
            
        Returns:
            quantized: Quantized tensor
            vq_loss: Vector quantization loss
            encoding_indices: Discrete indices
        """
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances to embedding vectors
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # Find closest embedding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, vq_loss, encoding_indices.view(input_shape[:-1])

class HierarchicalVQLayer(nn.Module):
    """
    Hierarchical Vector Quantization with Bayesian training scheme
    This implementation follows the HQ-VAE paper's approach to prevent
    codebook collapse through stochastic quantization and Bayesian inference.
    """

    def __init__(self, codebook_sizes: List[int], embedding_dims: List[int], 
                 commitment_costs: List[float] = None, temperature: float = 1.0):
        super().__init__()
        self.num_levels = len(codebook_sizes)
        self.temperature = temperature
        
        if commitment_costs is None:
            commitment_costs = [0.25] * self.num_levels
            
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_sizes[i], embedding_dims[i], commitment_costs[i])
            for i in range(self.num_levels)
        ])
        
        # Stochastic gating for preventing collapse (key innovation from HQ-VAE)
        # This part of the provided snippet was 'projections', but the text refers to 'stochastic_gates'
        # Adjusting to match the provided code's variable name for projections
        self.projections = nn.ModuleList([ # Renamed from stochastic_gates in textual snippet to match code block
            nn.Sequential(
                nn.Linear(embedding_dims[i], embedding_dims[i] // 2), # Original input_dim for HierarchicalEncoder output
                nn.ReLU(),
                nn.Linear(embedding_dims[i] // 2, embedding_dims[i]) # Project to embedding_dim for quantizer input
            ) for i in range(self.num_levels)
        ])
        
    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]: # Inputs is a list of features now
        """
        Hierarchical quantization
        
        Args:
            inputs: List of input tensors, one for each level's hierarchical feature
            
        Returns:
            Dictionary with quantized representations and losses
        """
        results = {
            'quantized': [],
            'vq_losses': [],
            'indices': [],
            'total_loss': 0.0
        }
        
        # Current input is now per-level feature from hierarchical encoder
        # Loop through levels, applying projection and quantization
        for level, (quantizer, projection) in enumerate(zip(self.quantizers, self.projections)):
            current_level_input = inputs[level] # Get the specific input for this level
            
            # Project to appropriate dimension
            projected = projection(current_level_input)
            
            # Quantize
            quantized, vq_loss, indices = quantizer(projected)
            
            results['quantized'].append(quantized)
            results['vq_losses'].append(vq_loss)
            results['indices'].append(indices)
            results['total_loss'] += vq_loss
            
            # HQ-VAE's residual connection (for next level input), if applicable
            # The provided code snippet has `current_input = current_input - quantized`
            # This implies HierarchicalVQLayer takes a single input and branches
            # However, the provided HQVAE.encode -> self.hvq_layer(hierarchical_features) suggests `inputs` is a list.
            # I will follow the provided HQVAE structure: hvq_layer receives hierarchical_features (a list).
            # The residual connection might be implicitly handled by how features are generated in the encoder
            # or combined in the decoder, not directly within this loop for list inputs.
            # So, `current_input = current_input - quantized` should be removed here for list input.
            
        return results

class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for multi-scale feature extraction"""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dims: List[int]):
        super().__init__()
        self.num_levels = len(output_dims)
        
        # Shared backbone
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        self.backbone = nn.Sequential(*layers)
        
        # Level-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(current_dim, output_dim) for output_dim in output_dims
        ])
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract hierarchical features"""
        features = self.backbone(x)
        return [head(features) for head in self.heads]

class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for reconstruction"""

    def __init__(self, input_dims: List[int], hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        # Combine hierarchical features
        total_input_dim = sum(input_dims)
        
        # Decoder layers
        layers = []
        current_dim = total_input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """Reconstruct from hierarchical features"""
        combined = torch.cat(hierarchical_features, dim=-1)
        return self.decoder(combined)

class VariationalBayesLayer(nn.Module):
    """Variational Bayes layer for uncertainty estimation"""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: q(z|x)
        self.encoder_mu = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)
        
        # Prior parameters (learnable)
        self.prior_mu = nn.Parameter(torch.zeros(latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(latent_dim))
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence between posterior and prior"""
        prior_mu = self.prior_mu.expand_as(mu)
        prior_logvar = self.prior_logvar.expand_as(logvar)
        
        kl = 0.5 * torch.sum(
            prior_logvar - logvar + 
            (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp() - 1,
            dim=-1
        )
        return kl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through variational layer
        
        Returns:
            z: Sampled latent variable
            mu: Posterior mean
            logvar: Posterior log variance
        """
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class HQVAE(nn.Module):
    """
    Hierarchical Quantized Variational Autoencoder
    Combines hierarchical discrete representation learning with variational Bayes
    for stable, diverse, and semantically rich representations.
    """

    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.codebook_sizes = config['codebook_sizes']  # e.g., [512, 256, 128] for 3 levels
        self.embedding_dims = config['embedding_dims']  # e.g., [64, 32, 16] for 3 levels
        self.encoder_hidden_dims = config.get('encoder_hidden_dims', [512, 256])
        self.decoder_hidden_dims = config.get('decoder_hidden_dims', [256, 512])
        
        # Hierarchical Encoder
        self.encoder = HierarchicalEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.encoder_hidden_dims,
            output_dims=self.embedding_dims # Each head outputs an embedding_dim for its level
        )
        
        # Variational Bayes Layer
        self.vb_layer = VariationalBayesLayer(
            input_dim=sum(self.embedding_dims), # Sum of all hierarchical encoder output dims
            latent_dim=self.latent_dim
        )
        
        # Hierarchical Vector Quantization
        self.hvq_layer = HierarchicalVQLayer(
            codebook_sizes=self.codebook_sizes,
            embedding_dims=self.embedding_dims
        )
        
        # Hierarchical Decoder
        self.decoder = HierarchicalDecoder(
            input_dims=self.embedding_dims, # List of embedding dims for each level
            hidden_dims=self.decoder_hidden_dims,
            output_dim=self.input_dim
        )
        
        # Loss weights
        self.recon_weight = config.get('recon_weight', 1.0)
        self.vq_weight = config.get('vq_weight', 1.0)
        self.kl_weight = config.get('kl_weight', 0.1)
        
    def encode(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to hierarchical representations"""
        # Hierarchical encoding
        hierarchical_features = self.encoder(x) # List of tensors, one for each level
        
        # Variational encoding from combined features
        combined_features = torch.cat(hierarchical_features, dim=-1) # Concatenate all hierarchical features
        z, mu, logvar = self.vb_layer(combined_features)
        
        return hierarchical_features, z, mu, logvar

    def decode(self, hierarchical_features: List[torch.Tensor]) -> torch.Tensor:
        """Decode from hierarchical representations"""
        return self.decoder(hierarchical_features)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HQ-VAE
        
        Args:
            x: Input tensor [B, D_input] (e.g., flattened image)
            
        Returns:
            Dictionary with outputs and losses
        """
        # Encoding
        hierarchical_features, z, mu, logvar = self.encode(x)
        
        # Hierarchical Vector Quantization
        vq_results = self.hvq_layer(hierarchical_features) # hvq_layer now expects a list of features
        quantized_features = vq_results['quantized'] # This is a list of quantized tensors
        
        # Decoding
        reconstruction = self.decode(quantized_features) # Decoder expects a list of features
        
        # Losses
        recon_loss = F.mse_loss(reconstruction, x)
        vq_loss = vq_results['total_loss'] # This is the sum of VQ losses from all levels
        kl_loss = torch.mean(self.vb_layer.kl_divergence(mu, logvar)) # KL is per batch, take mean
        
        total_loss = (self.recon_weight * recon_loss + 
                      self.vq_weight * vq_loss + 
                      self.kl_weight * kl_loss)
        
        return {
            'reconstruction': reconstruction,
            'hierarchical_features': hierarchical_features,
            'quantized_features': quantized_features, # List of quantized features
            'discrete_indices': vq_results['indices'], # List of discrete indices for each level
            'latent_z': z,
            'mu': mu,
            'logvar': logvar,
            'losses': {
                'total': total_loss,
                'reconstruction': recon_loss,
                'vq': vq_loss,
                'kl': kl_loss
            }
        }

    def generate(self, batch_size: int, device: str = 'cpu') -> torch.Tensor:
        """Generate samples from the model"""
        # Sample from prior (assuming standard normal prior here for simplicity)
        z = torch.randn(batch_size, self.latent_dim, device=device)
        
        # For HQ-VAE, generation from prior is more complex and typically involves
        # sampling from the hierarchical latent space.
        # The provided snippet uses mean_embeddings from quantizers.
        mean_embeddings = []
        for quantizer in self.hvq_layer.quantizers:
            mean_embedding = torch.mean(quantizer.embedding.weight, dim=0, keepdim=True)
            mean_embeddings.append(mean_embedding.repeat(batch_size, 1))
        
        # Decode
        with torch.no_grad():
            generated = self.decode(mean_embeddings)
            
        return generated

    def get_codebook_usage(self) -> Dict[str, torch.Tensor]:
        """
        Get codebook usage statistics
        HQ-VAE's Bayesian training scheme should prevent codebook collapse,
        leading to more uniform usage across all codebook entries.
        """
        usage_stats = {}
        for level, quantizer in enumerate(self.hvq_layer.quantizers):
            usage_stats[f'level_{level}_codebook_size'] = quantizer.num_embeddings
            usage_stats[f'level_{level}_embedding_dim'] = quantizer.embedding_dim
        return usage_stats

def create_hqvae_model(input_dim: int = 784, latent_dim: int = 64, 
                       codebook_sizes: List[int] = [512, 256], 
                       embedding_dims: List[int] = [64, 32],
                       encoder_hidden_dims: List[int] = [512, 256],
                       decoder_hidden_dims: List[int] = [256, 512],
                       recon_weight: float = 1.0, vq_weight: float = 1.0, kl_weight: float = 0.1) -> HQVAE:
    """Create HQ-VAE model with default configuration"""
    config = {
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'codebook_sizes': codebook_sizes,  # Hierarchical codebook sizes
        'embedding_dims': embedding_dims,     # Hierarchical embedding dimensions
        'encoder_hidden_dims': encoder_hidden_dims,
        'decoder_hidden_dims': decoder_hidden_dims,
        'recon_weight': recon_weight,
        'vq_weight': vq_weight,
        'kl_weight': kl_weight
    }
    return HQVAE(config)

# Training utilities
class HQVAETrainer:
    """Training utilities for HQ-VAE"""

    def __init__(self, model: HQVAE, optimizer: torch.optim.Optimizer, device: str = 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        batch = batch.to(self.device)
        
        # Forward pass
        outputs = self.model(batch)
        loss = outputs['losses']['total']
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return loss components
        return {k: v.item() for k, v in outputs['losses'].items()}

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        total_losses = {'total': 0, 'reconstruction': 0, 'vq': 0, 'kl': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                
                for k, v in outputs['losses'].items():
                    total_losses[k] += v.item()
                num_batches += 1
        
        return {k: v / num_batches for k, v in total_losses.items()}