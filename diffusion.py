# file: diffusion.py
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# --- (NEW) FOCAL LOSS FUNCTION ---
def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    Focal Loss function.
    
    Args:
        inputs (torch.Tensor): Logits of shape (B, K, H, W) where K is num_classes.
        targets (torch.Tensor): Ground truth labels of shape (B, H, W).
        alpha (float or torch.Tensor): Weighting factor for the rare class. Can be a float
                                       or a tensor of shape [K].
        gamma (float): Focusing parameter.
        reduction (str): 'mean', 'sum', or 'none'.
    
    Returns:
        torch.Tensor: The calculated focal loss.
    """
    # Calculate the cross-entropy loss without reduction
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    
    # Get the probabilities of the correct class
    pt = torch.exp(-ce_loss)
    
    # Create the focal loss term
    focal_term = (1 - pt) ** gamma
    
    # If alpha is a float, convert to a tensor for indexing
    if isinstance(alpha, (float, int)):
        alpha_t = torch.tensor([1 - alpha, alpha], device=inputs.device) # Assume alpha is for class 1
    else:
        alpha_t = alpha

    # Gather the alpha weights for each pixel based on its target class
    alpha_t = alpha_t.gather(0, targets.flatten()).view_as(targets)
    
    # Calculate the final focal loss
    loss = alpha_t * focal_term * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

# --- (MODIFIED) DiscreteDiffusion CLASS ---
class DiscreteDiffusion:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, num_classes=2, focal_loss_gamma=2.0, focal_loss_alpha=0.9):
        self.num_timesteps = timesteps
        self.num_classes = num_classes
        
        # --- (NEW) Store focal loss parameters ---
        self.gamma = focal_loss_gamma
        # Alpha is the weight for the positive class (1). We construct the weight tensor.
        self.alpha = torch.tensor([1.0 - focal_loss_alpha, focal_loss_alpha])
        
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        
        q_one_step = torch.zeros(timesteps, num_classes, num_classes)
        q_one_step[:, 0, 0] = 1 - self.betas / 2
        q_one_step[:, 0, 1] = self.betas / 2
        q_one_step[:, 1, 0] = self.betas / 2
        q_one_step[:, 1, 1] = 1 - self.betas / 2
        self.q_one_step = q_one_step

        self.q_bar = torch.zeros(timesteps, num_classes, num_classes)
        q_bar_t = torch.eye(num_classes)
        for t in range(timesteps):
            q_bar_t = torch.matmul(q_bar_t, self.q_one_step[t])
            self.q_bar[t] = q_bar_t

    def to(self, device):
        self.betas = self.betas.to(device)
        self.q_one_step = self.q_one_step.to(device)
        self.q_bar = self.q_bar.to(device)
        # --- (NEW) Move alpha tensor to the correct device ---
        self.alpha = self.alpha.to(device)
        return self

    def q_sample(self, x_start, t):
      batch_size, channels, height, width = x_start.shape
      num_pixels = channels * height * width
      t_expanded = t.repeat_interleave(num_pixels)
      q_bar_t_expanded = self.q_bar[t_expanded]
      x_start_pixels = x_start.long().flatten()
      pixel_indices = torch.arange(len(x_start_pixels), device=x_start.device)
      probs = q_bar_t_expanded[pixel_indices, x_start_pixels]
      noisy_pixels = torch.multinomial(probs, num_samples=1).squeeze(1)
      return noisy_pixels.view(x_start.shape).float()

    def q_posterior_logits(self, x_t, x_0, t):
      B, C, H, W = x_t.shape
      num_pixels = C * H * W
      q_t_batch = self.q_one_step[t]
      q_t_expanded = q_t_batch.repeat_interleave(num_pixels, dim=0)
      xt_pixels = x_t.long().flatten()
      fact1_probs = q_t_expanded[torch.arange(len(xt_pixels)), :, xt_pixels]
      
      t_minus_1 = t - 1
      q_bar_t_minus_1_batch = torch.where(
          t_minus_1.view(-1, 1, 1) < 0,
          torch.eye(self.num_classes, device=x_t.device).expand(B, -1, -1),
          self.q_bar[t_minus_1]
      )
      q_bar_t_minus_1_expanded = q_bar_t_minus_1_batch.repeat_interleave(num_pixels, dim=0)
      x0_pixels = x_0.long().flatten()
      fact2_probs = q_bar_t_minus_1_expanded[torch.arange(len(x0_pixels)), x0_pixels, :]
      
      logits_flat = torch.log(fact1_probs + 1e-20) + torch.log(fact2_probs + 1e-20)
      return logits_flat.view(B, num_pixels, self.num_classes).permute(0, 2, 1).view(B, self.num_classes, H, W)

    def compute_loss(self, model, x_start, condition_image=None):
        """ Computes the loss: prediction of x_0 from x_t. """
        b, c, h, w = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        x_t = self.q_sample(x_start, t)
        
        if condition_image is not None:
            predicted_x0_logits = model(x_t, t, condition_image)
        else:
            predicted_x0_logits = model(x_t, t)
            
        target = x_start.long().squeeze(1)
        
        # --- (MODIFIED) USE FOCAL LOSS INSTEAD OF CROSS ENTROPY ---
        loss = focal_loss(
            predicted_x0_logits, 
            target, 
            alpha=self.alpha, 
            gamma=self.gamma
        )
        return loss

    @torch.no_grad()
    def p_sample(self, model, x_t, t_tensor, condition_image=None):
        """ Sample x_{t-1} from the model's prediction of x_0. """
        model.eval()
        
        if condition_image is not None:
            predicted_x0_logits = model(x_t, t_tensor, condition_image)
        else:
            predicted_x0_logits = model(x_t, t_tensor)

        predicted_x0_probs = F.softmax(predicted_x0_logits, dim=1)
        predicted_x0 = torch.argmax(predicted_x0_probs, dim=1).unsqueeze(1)

        posterior_logits = self.q_posterior_logits(x_t, predicted_x0, t_tensor)
        posterior_probs = F.softmax(posterior_logits, dim=1)
        
        b, k, h, w = posterior_probs.shape
        sampled_pixels = torch.multinomial(posterior_probs.permute(0, 2, 3, 1).reshape(-1, k), 1).squeeze(1)
        
        return sampled_pixels.view(b, h, w).unsqueeze(1).float()

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=1, channels=1, condition_image=None):
        """ Generates new images by running the full reverse diffusion process. """
        device = next(model.parameters()).device
        img = torch.randint(0, self.num_classes, (batch_size, channels, image_size, image_size), device=device).float()
        
        if condition_image is not None:
            if batch_size != condition_image.shape[0]:
                print(f"Warning: batch_size mismatch. Adjusting from {batch_size} to {condition_image.shape[0]}")
                batch_size = condition_image.shape[0]

        for i in tqdm(reversed(range(self.num_timesteps)), desc="Sampling", total=self.num_timesteps):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, condition_image)
            
        return img