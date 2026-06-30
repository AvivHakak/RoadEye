import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
from roboflow import Roboflow
import requests

def download_dataset():
    api_key = "gSYbQ4Mn1KcEImEvvD6G"
    response = requests.get(f"https://api.roboflow.com/?api_key={api_key}")
    data = response.json()
    my_workspace = data['workspace']
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(my_workspace).project("potholes-detection-d4rma-voqpa")
    dataset = project.version(1).download("yolov8")
    return dataset.location

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

class RoadEyeGridModel(nn.Module):
    def __init__(self):
        super(RoadEyeGridModel, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = mobilenet.features
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.detector = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 5, kernel_size=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.detector(features)
        return out

class RoadEyeLoss(nn.Module):
    def __init__(self):
        super(RoadEyeLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.bce = nn.BCELoss(reduction="sum")
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5

    def forward(self, predictions, target):
        obj = target[:, 0, ...] == 1
        noobj = target[:, 0, ...] == 0
        noobj_loss = self.bce(predictions[:, 0, ...][noobj], target[:, 0, ...][noobj])

        if obj.sum() > 0:
            obj_loss = self.bce(predictions[:, 0, ...][obj], target[:, 0, ...][obj])
            pred_boxes = predictions[:, 1:5, ...].permute(0, 2, 3, 1)
            target_boxes = target[:, 1:5, ...].permute(0, 2, 3, 1)
            box_loss = self.mse(pred_boxes[obj], target_boxes[obj])
        else:
            obj_loss = 0.0
            box_loss = 0.0

        total_loss = self.lambda_coord * box_loss + obj_loss + self.lambda_noobj * noobj_loss
        return total_loss

if __name__ == '__main__':
    dataset_path = download_dataset()
    train_dataset = RoadEyeGridDataset(img_dir=f'{dataset_path}/train/images', label_dir=f'{dataset_path}/train/labels')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    model = RoadEyeGridModel().to(device)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = RoadEyeLoss()
    
    num_epochs = 10
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            predictions = model(images)
            loss = criterion(predictions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
                
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), 'roadeye_grid_model_new.pt')
    print("Model saved locally as roadeye_grid_model_new.pt")