# file: dataset_car.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import random

class CarPointDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, image_size=64, transform=None, augment=True):
        """
        Dataset for car images and their corresponding point masks.
        
        Args:
            img_dir (str): Directory with all the images.
            mask_dir (str): Directory with all the point masks.
            file_list (list): A list of filenames to be loaded by this dataset instance.
            image_size (int): The target image size. Assumes images are pre-resized.
            transform (callable, optional): Optional transform for the condition image.
            augment (bool): Whether to apply random horizontal flipping.
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        
        # This dataset instance will only use the files passed in file_list
        self.img_files = file_list
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transform
            
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = self.img_files[index]
        base_name = os.path.splitext(img_name)[0]
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, f"{base_name}.png")

        cond_image = Image.open(img_path).convert('RGB')
        point_mask = Image.open(mask_path).convert('L')

        if self.augment and random.random() > 0.5:
            cond_image = cond_image.transpose(Image.FLIP_LEFT_RIGHT)
            point_mask = point_mask.transpose(Image.FLIP_LEFT_RIGHT)

        cond_image_tensor = self.transform(cond_image)
        
        dot_map_tensor = self.mask_transform(point_mask)
        dot_map_tensor = (dot_map_tensor > 0.5).float()
        
        return cond_image_tensor, dot_map_tensor
    

# #visualization function for debugging
# def visualize_dataset(dataset, num_samples=5):
#     import matplotlib.pyplot as plt
    
#     fig, axes = plt.subplots(num_samples, 2, figsize=(10, 2 * num_samples))
    
#     for i in range(num_samples):
#         cond_image, dot_map = dataset[i]
#         cond_image = cond_image.permute(1, 2, 0).numpy()
#         dot_map = dot_map.squeeze().numpy()
        
#         axes[i, 0].imshow(cond_image)
#         axes[i, 0].set_title('Condition Image')
#         axes[i, 0].axis('off')
        
#         axes[i, 1].imshow(dot_map, cmap='gray')
#         axes[i, 1].set_title('Point Mask')
#         axes[i, 1].axis('off')
    
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     # Example usage
#     img_dir = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\Carvana_image_segmentation_dataset\processed_train_images_10_points"
#     mask_dir = r"C:\Users\Mehmet_Postdoc\Desktop\datasets_for_experiments\Carvana_image_segmentation_dataset\processed_train_point_masks_10_points"
#     file_list = os.listdir(img_dir)  # Assuming all files in img_dir are valid images
    
#     dataset = CarPointDataset(img_dir, mask_dir, file_list, image_size=64)
    
#     # Visualize some samples
#     visualize_dataset(dataset, num_samples=9)