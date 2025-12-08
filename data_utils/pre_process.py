import os
import torchvision
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np 
import albumentations as A

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 2.0] = 0.0
    mask[(mask == 1.0) | (mask == 3.0)] = 1.0
    return mask
class ImageSegmentationDataset(Dataset):
    def __init__(self, image_list, image_labels, transforms):
        super().__init__()
        self.images = image_list
        self.labels = image_labels
        self.transforms = transforms
    
    def __getitem__(self, index):
        img = self.images[index]
        label = preprocess_mask(self.labels[index])

        if self.transforms:
            transformed = self.transforms(image = img, mask = label)
            img = transformed["image"]
            label = transformed["mask"]

        return img, label
    
    def __len__(self):
        return len(self.images)

def load_images_from_folder(folder_path):
    img_paths = sorted([f"{folder_path}/{x}" for x in os.listdir(folder_path) if ('.jpg' in x) or ('.png' in x)])

    images = [np.asarray(Image.open(x)) for x in img_paths]
    return images

def load_datasets(input_folder_path, label_folder_path, train_transforms, val_test_transforms):
    # Load input data
    input_images = load_images_from_folder(input_folder_path)

    # Load in labels
    label_images = load_images_from_folder(label_folder_path)
    label_images = [x.long() for x in label_images]

    # Create Train, Val and Test Split
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    input_images, label_images, test_size=0.3, random_state=42, shuffle=True
    )

    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=0.5, random_state=42, shuffle=True
    )

    # Create Train, Val and Test Torch Dataset
    train_dataset = ImageSegmentationDataset(train_imgs, train_labels, train_transforms)
    val_dataset = ImageSegmentationDataset(val_imgs, val_labels, val_test_transforms)
    test_dataset = ImageSegmentationDataset(test_imgs, test_labels, val_test_transforms)

    return train_dataset, val_dataset, test_dataset