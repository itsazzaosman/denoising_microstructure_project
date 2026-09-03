import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class EBSD_Dataset(Dataset):
    def __init__(self, clean_dir, noisy_dir):
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.image_filenames = sorted(os.listdir(clean_dir))
        
        self.transform = transforms.Compose([
            transforms.ToTensor() # Converts 0-255 RGB to 0.0-1.0 float tensor
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        
        clean_path = os.path.join(self.clean_dir, img_name)
        noisy_path = os.path.join(self.noisy_dir, img_name)
        
        clean_img = Image.open(clean_path).convert("RGB")
        noisy_img = Image.open(noisy_path).convert("RGB")
        
        clean_tensor = self.transform(clean_img)
        noisy_tensor = self.transform(noisy_img)
        
        return noisy_tensor, clean_tensor