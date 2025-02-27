import torch
import torch.nn as nn
import torch.nn.functional as F
from drakes_dna.diffusion import DNADiffusion  # Base class from DRAKES

class DDPPDNA(DNADiffusion):
    def __init__(self, 
                 base_model,
                 reward_model,
                 beta_schedule,
                 kl_weight=0.1,
                 num_samples=64,
                 temp=0.1):
        super().__init__(base_model, reward_model, beta_schedule)
        self.kl_weight = kl_weight
        self.num_samples = num_samples
        self.temp = temp
        
    def compute_target_distribution(self, logits_base, rewards):
        """DDPP importance sampling target calculation"""
        log_weights = rewards / self.temp
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        weights = torch.exp(log_weights)
        
        # Expand weights to match logits dimensions
        weights = weights.unsqueeze(-1).expand_as(logits_base)
        target_dist = (logits_base.softmax(-1) * weights).sum(1)
        return target_dist

    def compute_loss(self, x0):
        batch_size, seq_len = x0.shape
        device = x0.device
        
        # Sample timestep and mask input
        t = torch.randint(0, len(self.beta), (batch_size,))
        mask = self._apply_masking(x0, t)
        xt = torch.where(mask, self.mask_token, x0)
        
        # Get base model predictions
        with torch.no_grad():
            logits_base = self.base_model(xt, t)
            
        # Importance sampling from base model
        samples = torch.distributions.Categorical(
            logits=logits_base
        ).sample((self.num_samples,))  # [num_samples, batch_size, seq_len]
        
        # Compute rewards for all samples
        flat_samples = samples.reshape(-1, seq_len)
        rewards = self.reward_model(flat_samples).view(self.num_samples, batch_size)
        
        # Compute DDPP target distribution
        target_dist = self.compute_target_distribution(logits_base, rewards)
        
        # Get current model predictions
        logits_current = self(xt, t)
        current_dist = logits_current.log_softmax(-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(
            current_dist, 
            target_dist.log(),
            reduction='batchmean',
            log_target=True
        )
        
        # Reward maximization term
        current_samples = torch.distributions.Categorical(
            logits=logits_current
        ).sample((self.num_samples,))
        current_rewards = self.reward_model(current_samples.reshape(-1, seq_len))
        reward_loss = -current_rewards.mean()
        
        # Combined loss
        total_loss = reward_loss + self.kl_weight * kl_loss
        
        return total_loss

# Modified training loop additions
class DDPPTrainer:
    def __init__(self, 
                 model,
                 optimizer,
                 temp_scheduler=None,
                 kl_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.temp_scheduler = temp_scheduler
        self.kl_scheduler = kl_scheduler
        
    def train_step(self, batch):
        self.optimizer.zero_grad()
        loss = self.model.compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        
        # Dynamic temperature annealing
        if self.temp_scheduler:
            self.model.temp = self.temp_scheduler.step()
            
        # KL weight scheduling
        if self.kl_scheduler:
            self.model.kl_weight = self.kl_scheduler.step()
            
        return loss.item()
