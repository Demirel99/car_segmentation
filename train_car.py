# file: train_car.py
import torch
import os
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import time
import random

from model import ConditionalDiffusionModel64
from diffusion import DiscreteDiffusion
from dataset_car import CarPointDataset

def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    sample_dir = os.path.join(args.save_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- Data Loading and Splitting ---
    print(f"Loading data from {args.img_dir} and {args.mask_dir}")
    all_files = sorted([f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    # Reproducible train/validation split
    random.seed(args.seed)
    random.shuffle(all_files)
    
    split_idx = int(len(all_files) * (1 - args.val_split))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Total files: {len(all_files)}. Training on {len(train_files)}, Validating on {len(val_files)}.")

    train_dataset = CarPointDataset(
        img_dir=args.img_dir, 
        mask_dir=args.mask_dir, 
        file_list=train_files,
        image_size=args.img_size,
        augment=True
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = CarPointDataset(
        img_dir=args.img_dir, 
        mask_dir=args.mask_dir,
        file_list=val_files, 
        image_size=args.img_size,
        augment=False
    )
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True) 
    
    # --- Model, Diffusion, and Optimizer Setup ---
    model = ConditionalDiffusionModel64(vgg_pretrained=True).to(device)
    
    print(f"Using Focal Loss with gamma={args.gamma} and alpha={args.alpha}")
    diffusion = DiscreteDiffusion(
        timesteps=args.timesteps,
        num_classes=2,
        focal_loss_gamma=args.gamma,
        focal_loss_alpha=args.alpha
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Get fixed validation batch for sampling visualization ---
    try:
        fixed_val_batch = next(iter(val_dataloader))
        fixed_cond_images, fixed_true_dots = fixed_val_batch[0].to(device), fixed_val_batch[1]
        print(f"Loaded a fixed validation batch of size {fixed_cond_images.shape[0]} for sampling.")
    except StopIteration:
        print("Validation set is empty or too small. Cannot create fixed validation batch for sampling.")
        fixed_cond_images, fixed_true_dots = None, None

    # --- Training Loop ---
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        model.train()
        
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (condition_image, x_start) in enumerate(pbar):
            optimizer.zero_grad()
            condition_image, x_start = condition_image.to(device), x_start.to(device)
            loss = diffusion.compute_loss(model, x_start, condition_image)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            
            if fixed_cond_images is not None:
                model.eval()
                with torch.no_grad():
                    generated_samples = diffusion.sample(
                        model, image_size=args.img_size, batch_size=fixed_cond_images.shape[0], 
                        condition_image=fixed_cond_images
                    ).cpu()
                
                mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
                cond_rgb = fixed_cond_images.cpu() * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
                
                true_dots_rgb = fixed_true_dots.repeat(1, 3, 1, 1)
                generated_rgb = generated_samples.repeat(1, 3, 1, 1)
                
                comparison_grid = torch.cat([cond_rgb, true_dots_rgb, generated_rgb], dim=3)
                sample_path = os.path.join(sample_dir, f"sample_epoch_{epoch+1}.png")
                save_image(comparison_grid, sample_path, nrow=1, normalize=False)
                print(f"Saved checkpoint and samples for epoch {epoch+1}")

    print("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model for car point localization.")
    parser.add_argument('--img_dir', type=str, required=True, help='Path to the processed images directory.')
    parser.add_argument('--mask_dir', type=str, required=True, help='Path to the processed point masks directory.')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--save_dir', type=str, default='results_car_points', help='Directory to save results.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save checkpoint and samples every N epochs.')
    parser.add_argument('--timesteps', type=int, default=200, help='Number of diffusion timesteps.')
    parser.add_argument('--val_split', type=float, default=0.2, help='Proportion of the dataset to use for validation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible train/val split.')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss focusing parameter (gamma).')
    parser.add_argument('--alpha', type=float, default=0.95, help='Focal loss alpha parameter (weight for the positive class).')
    args = parser.parse_args()
    train(args)
    