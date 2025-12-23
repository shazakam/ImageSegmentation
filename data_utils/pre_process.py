import os
import torchvision
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np 
import albumentations as A
from pathlib import Path
import torch
def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[(mask == 2.0)] = 0.0
    mask[(mask == 1.0)| (mask == 3.0)] = 1.0
    return mask
class ImageSegmentationDataset(Dataset):
    def __init__(self, image_path_list, image_path_labels, transforms):
        super().__init__()
        self.images = image_path_list
        self.labels = image_path_labels
        self.transforms = transforms
    
    def __getitem__(self, index):
        img = np.asarray(Image.open(self.images[index]).convert("RGB"))
        label_img = np.asarray(Image.open(self.labels[index]))
        label = preprocess_mask(label_img)

        if self.transforms:
            transformed = self.transforms(image = img, mask = label)
            img = transformed["image"]
            label = torch.unsqueeze(transformed["mask"], dim = 0)

        return img, label
    
    def __troubleshoot__(self, index):
        img = np.asarray(Image.open(self.images[index]).convert("RGB"))
        label_img = np.asarray(Image.open(self.labels[index]).convert("RGB"))
        label = preprocess_mask(label_img)

        if self.transforms:
            transformed = self.transforms(image = img, mask = label)
            img = transformed["image"]
            label = torch.unsqueeze(transformed["mask"], dim = 0)

        return img, label, self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)

def load_image_paths_from_folder(folder_path):
    img_paths = sorted([f"{folder_path}/{x}" for x in os.listdir(folder_path) if ('.jpg' in x) or ('.png' in x)])

    return img_paths

def load_image_path_splits(input_images_paths, label_images_paths, shuffle):
    # Create Train, Val and Test Split
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    input_images_paths, label_images_paths, test_size=0.3, random_state=42, shuffle=shuffle
    )

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=42, shuffle=shuffle
    )

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_labels)

def load_images_and_labels(input_folder_path, label_folder_path):
    # Load input data
    input_images = load_image_paths_from_folder(input_folder_path)

    # Load in label
    label_images = load_image_paths_from_folder(label_folder_path)

    return input_images, label_images

# Change this to just save the image paths instead in a txt file
def save_image_paths(image_list, label_list, train_val_test, save_folder = "saved_data"):
    img_dir = Path(f"{save_folder}/{train_val_test}/images")
    mask_dir = Path(f"{save_folder}/{train_val_test}/masks")

    # Create directories if they don't exist
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    with open(f"{img_dir}/image_paths.txt", "w") as f:
        for idx, image in enumerate(image_list):
            f.write(image + "\n")

    with open(f"{mask_dir}/label_paths.txt", "w") as f:
        for idx, label in enumerate(label_list):
            f.write(label+ "\n")

def load_datasets(input_images, label_images, train_transforms, val_test_transforms, save_path = None, shuffle = True):

    # Create Train, Val and Test Split
    (train_imgs, train_labels), (val_imgs, val_labels),  (test_imgs, test_labels) = load_image_path_splits(input_images, 
                                                                                                    label_images,
                                                                                                    shuffle)
    if save_path != None:
        save_image_paths(train_imgs, train_labels, "train", save_path)
        save_image_paths(val_imgs, val_labels, "validation", save_path)
        save_image_paths(test_imgs, test_labels, "test", save_path)

    # Create Train, Val and Test Torch Dataset
    train_dataset = ImageSegmentationDataset(train_imgs, train_labels, train_transforms)
    val_dataset = ImageSegmentationDataset(val_imgs, val_labels, val_test_transforms)
    test_dataset = ImageSegmentationDataset(test_imgs, test_labels, val_test_transforms)

    return train_dataset, val_dataset, test_dataset