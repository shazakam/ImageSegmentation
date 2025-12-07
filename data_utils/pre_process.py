import os
import torchvision
from torch.utils.data import Dataset, TensorDataset
from PIL import Image
from torchvision import transforms

class ImageSegmentationDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return 0

def load_images_from_folder(folder_path):
    img_paths = sorted([f"{folder_path}/{x}" for x in os.listdir(folder_path) if ('.jpg' in x) or ('.png' in x)])
    convert_to_tensor = transforms.ToTensor()
    images = [convert_to_tensor(Image.open(x)) for x in img_paths]
    return images

def load_datasets(input_folder_path, label_folder_path, input_transforms):
    # Load input data
    input_images = load_images_from_folder(input_folder_path)

    # Load in labels
    label_images = load_images_from_folder(label_folder_path)

    # Create Train, Val and Test Split

    # Create Train, Val and Test Torch Dataset

    return