import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class RoadEyeGridDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, grid_size=7):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.grid_size = grid_size
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        
        label_name = img_name.rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        target = torch.zeros((5, self.grid_size, self.grid_size), dtype=torch.float32)
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id, x, y, w, h = map(float, parts)
                        
                        grid_x = int(x * self.grid_size)
                        grid_y = int(y * self.grid_size)
                        
                        if grid_x == self.grid_size: grid_x -= 1
                        if grid_y == self.grid_size: grid_y -= 1
                        
                        if target[0, grid_y, grid_x] == 0:
                            target[0, grid_y, grid_x] = 1.0
                            target[1, grid_y, grid_x] = x
                            target[2, grid_y, grid_x] = y
                            target[3, grid_y, grid_x] = w
                            target[4, grid_y, grid_x] = h
        
        return image, target