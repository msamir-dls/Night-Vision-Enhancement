import torch
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class LowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.low_light_dir = os.path.join(root_dir, "low")
        self.high_light_dir = os.path.join(root_dir, "high")
        self.transform = transform


        self.low_images = sorted(os.listdir(self.low_light_dir))
        self.high_images = sorted(os.listdir(self.high_light_dir))

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
      
        low_path = os.path.join(self.low_light_dir, self.low_images[idx])
        high_path = os.path.join(self.normal_light_dir, self.high_images[idx])

   
        low_img = cv2.imread(low_path)
        high_img = cv2.imread(high_path)


        low_img = cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB)
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB)

  
        if self.transform:
            low_img = self.transform(low_img)
            normal_img = self.transform(high_img)

        return low_img, high_img 


transform = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

train_images_path = r"lol_dataset/our485"
val_images_path = r"lol_dataset/eval15/"

train_dataset = LowLightDataset(root_dir=train_images_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
train_low, train_high = next(iter(train_dataloader))
print(f"Low-Light Image Shape: {train_low.shape}, Normal Image Shape: {train_high.shape}")

val_dataset = LowLightDataset(root_dir=val_images_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
val_low, val_high = next(iter(val_dataloader))
print(f"Low-Light Image Shape: {val_low.shape}, Normal Image Shape: {val_high.shape}")

