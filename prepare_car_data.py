# file: prepare_car_data.py
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import shutil

def generate_and_save_point_masks(train_img_dir, train_mask_dir, output_img_dir, output_point_mask_dir, num_points=10, image_size=64):
    """
    Processes a dataset of images and masks to generate sparse point masks.

    For each image in the input directory, it:
    1. Resizes and copies the original image to the output image directory.
    2. Generates a sparse point mask from the corresponding full mask.
    3. Saves this new point mask as a PNG image in the output mask directory.

    Args:
        train_img_dir (str): Path to the directory with original training images.
        train_mask_dir (str): Path to the directory with original training masks.
        output_img_dir (str): Path to the directory where processed images will be saved.
        output_point_mask_dir (str): Path to the directory where generated point masks will be saved.
        num_points (int): The number of points to sample from each mask.
        image_size (int): The size to resize images and masks to.
    """
    # --- 1. Setup ---
    print("Starting data preprocessing...")
    
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_point_mask_dir, exist_ok=True)
    print(f"Output images will be saved to: {output_img_dir}")
    print(f"Output point masks will be saved to: {output_point_mask_dir}")

    image_files = sorted(os.listdir(train_img_dir))
    mask_files = sorted(os.listdir(train_mask_dir))

    if len(image_files) != len(mask_files):
        print("Error: The number of images and masks does not match!")
        return

    transform_mask = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    transform_image = transforms.Resize((image_size, image_size))

    # --- 2. Processing Loop ---
    for i in tqdm(range(len(image_files)), desc="Processing Images"):
        img_name = image_files[i]
        mask_name = mask_files[i]

        input_img_path = os.path.join(train_img_dir, img_name)
        input_mask_path = os.path.join(train_mask_dir, mask_name)

        mask = Image.open(input_mask_path).convert("L")
        mask_tensor = transform_mask(mask)

        foreground_coords = torch.nonzero(mask_tensor > 0.5)
        point_mask_tensor = torch.zeros_like(mask_tensor)

        if len(foreground_coords) > 0:
            foreground_coords = foreground_coords[:, 1:]
            num_foreground_pixels = foreground_coords.shape[0]
            random_indices = torch.randint(0, num_foreground_pixels, (num_points,))
            sampled_points = foreground_coords[random_indices]
            point_mask_tensor[0, sampled_points[:, 0], sampled_points[:, 1]] = 1.0

        # --- 3. Save the results ---
        # a) Resize and save the original image
        img = Image.open(input_img_path).convert("RGB")
        img_resized = transform_image(img)
        output_img_path = os.path.join(output_img_dir, img_name)
        img_resized.save(output_img_path)

        # b) Save the point mask
        point_mask_np = point_mask_tensor.squeeze(0).mul(255).byte().cpu().numpy()
        point_mask_image = Image.fromarray(point_mask_np, mode='L')
        
        base_filename = os.path.splitext(img_name)[0]
        output_mask_path = os.path.join(output_point_mask_dir, f"{base_filename}.png")
        point_mask_image.save(output_mask_path)

    print("\nPreprocessing complete!")
    print(f"Total images processed: {len(image_files)}")


if __name__ == '__main__':
    # --- USER-DEFINED PATHS ---
    ROOT_DATA_PATH = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\Carvana_image_segmentation_dataset"
    
    # 1. DEFINE YOUR INPUT DIRECTORIES
    TRAIN_IMG_DIR = os.path.join(ROOT_DATA_PATH, "train")
    TRAIN_MASK_DIR = os.path.join(ROOT_DATA_PATH, "train_masks")

    # 2. DEFINE YOUR OUTPUT DIRECTORIES
    OUTPUT_IMG_DIR = os.path.join(ROOT_DATA_PATH, "processed_train_images_10_points")
    OUTPUT_POINT_MASK_DIR = os.path.join(ROOT_DATA_PATH, "processed_train_point_masks_10_points")

    # 3. DEFINE PARAMETERS
    NUM_POINTS = 10
    IMAGE_SIZE = 64

    # 4. RUN THE SCRIPT
    generate_and_save_point_masks(
        train_img_dir=TRAIN_IMG_DIR,
        train_mask_dir=TRAIN_MASK_DIR,
        output_img_dir=OUTPUT_IMG_DIR,
        output_point_mask_dir=OUTPUT_POINT_MASK_DIR,
        num_points=NUM_POINTS,
        image_size=IMAGE_SIZE
    )