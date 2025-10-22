import os
import shutil
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from numpy import array
from utils.config import Config
import kagglehub

class MapDataset(Dataset):
    def __init__(self, high_res_dir, low_res_dir, img_size):
        super().__init__()
        self.high_res_dir = high_res_dir
        self.low_res_dir = low_res_dir
        self.high_res_images = os.listdir(high_res_dir)
        self.low_res_images = os.listdir(low_res_dir)
        self.img_size = img_size

    def __len__(self):
        return len(max(self.high_res_images, self.low_res_images))
    
    def __getitem__(self, idx):
        high_res_img_path = os.path.join(self.high_res_dir, self.high_res_images[idx])
        low_res_img_path = os.path.join(self.low_res_dir, self.low_res_images[idx])

        high_res_img = Image.open(high_res_img_path).convert("RGB")
        low_res_img = Image.open(low_res_img_path).convert("RGB")
        
        # high_res_img = array(high_res_img)
        # low_res_img = array(low_res_img)

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        high_res_img = transform(high_res_img)
        low_res_img = transform(low_res_img)

        return high_res_img, low_res_img


if __name__ == "__main__":
    config = Config()

    # 1. Download dataset (cached in kagglehub's default path)
    path = kagglehub.dataset_download(config.DATASET_NAME)
    print("KaggleHub cache path:", path)

    # 2. Define target path in your current working directory
    target_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(target_dir, exist_ok=True)

    # 3. Copy all contents from kagglehub cache to target directory
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(target_dir, item)

        if os.path.isdir(src):
            # Copy folder
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            # Copy file
            shutil.copy2(src, dst)

    print(f"Dataset copied to: {target_dir}")
