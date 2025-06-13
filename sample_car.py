# file: sample_car.py
import torch
import os
import argparse
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

from model import ConditionalDiffusionModel64
from diffusion import DiscreteDiffusion

def sample(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    model = ConditionalDiffusionModel64(vgg_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    diffusion = DiscreteDiffusion(timesteps=args.timesteps, num_classes=2).to(device)

    transform_cond = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cond_image = Image.open(args.cond_img).convert("RGB")
    cond_tensor = transform_cond(cond_image).unsqueeze(0).to(device)
    
    print(f"Generating {args.num_samples} samples for {args.cond_img}...")
    all_samples = []
    with torch.no_grad():
        for _ in tqdm(range(args.num_samples), desc="Generating Samples"):
            generated_sample = diffusion.sample(
                model, image_size=args.img_size, batch_size=1, condition_image=cond_tensor
            ).cpu()
            all_samples.append(generated_sample)

    # --- MODIFICATION START: Use logical OR (union) instead of mean ---
    print("\nAggregating samples using union (logical OR) operation...")
    stacked_samples = torch.cat(all_samples, dim=0)
    
    # Create a union map where a pixel is 1 if it's 1 in ANY of the generated samples.
    # This is equivalent to a logical OR across all samples.
    union_map = (torch.sum(stacked_samples, dim=0, keepdim=True) > 0).float()

    # --- Post-process the union map to find final points ---
    # Convert tensor to numpy for scipy processing
    union_map_np = union_map.squeeze().numpy()

    # Find connected components (blobs) in the union map
    labeled_array, num_features = ndimage.label(union_map_np)
    print(f"Found {num_features} blobs in the union map.")

    # Find the geometric center of each labeled blob
    # Note: Since the input `union_map_np` is binary, center_of_mass finds the geometric center.
    centers = ndimage.center_of_mass(union_map_np, labeled_array, range(1, num_features + 1))
    
    # Create the final dot map by placing a single point at each blob's center
    final_dot_map_np = np.zeros((args.img_size, args.img_size), dtype=np.float32)
    for i, (y, x) in enumerate(centers):
        y_int, x_int = int(round(y)), int(round(x))
        if 0 <= y_int < args.img_size and 0 <= x_int < args.img_size:
            final_dot_map_np[y_int, x_int] = 1.0

    final_dot_map_tensor = torch.from_numpy(final_dot_map_np).unsqueeze(0).unsqueeze(0)
    # --- MODIFICATION END ---


    # --- Create Visualization Grid ---
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    cond_image_vis = cond_tensor.cpu() * std.view(1, 3, 1, 1) + mean.view(1, 3, 1, 1)
    
    images_to_show = [cond_image_vis]
    
    # Add Ground Truth if provided
    if args.gt_mask:
        gt_mask_pil = Image.open(args.gt_mask).convert('L')
        transform_gt = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
        ])
        true_dot_map = (transform_gt(gt_mask_pil) > 0.5).float().unsqueeze(0)
        images_to_show.append(true_dot_map.repeat(1, 3, 1, 1))

    # Add the Union Map and the Final Predicted Dots to the visualization
    images_to_show.extend([
        union_map.repeat(1, 3, 1, 1),
        final_dot_map_tensor.repeat(1, 3, 1, 1)
    ])

    # Concatenate images horizontally. Titles (from left to right):
    # Cond Image, [Opt: GT Mask], Union Map, Final Prediction
    comparison_grid = torch.cat(images_to_show, dim=3) 
    save_image(comparison_grid, args.out, normalize=False)
    print(f"\nSaved visualization grid to {args.out}")
    print(f"Final Count: {num_features}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate point map samples from a trained car localization model.")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument('--cond_img', type=str, required=True, help="Path to the conditional image.")
    parser.add_argument('--gt_mask', type=str, default=None, help="(Optional) Path to the ground truth point mask (.png).")
    parser.add_argument('--out', type=str, default="generated_car_sample.png", help="Output filename for the visualization grid.")
    parser.add_argument('--img_size', type=int, default=64, help='Image size (must match model).')
    parser.add_argument('--timesteps', type=int, default=200, help="Number of timesteps (must match training).")
    parser.add_argument('--num_samples', type=int, default=1000, help="Number of samples to generate and create a union from.")
    # --- MODIFICATION: Removed threshold argument as it's no longer used ---
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sample(args)
