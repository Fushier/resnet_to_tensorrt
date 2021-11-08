from torchvision import transforms
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import os


class TestImageDataset(Dataset):
    def __init__(self,root):
      imgs=os.listdir(root)
      self.imgs=[os.path.join(root,k) for k in imgs]
      self.transforms=transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
      img_path = self.imgs[index]
      pil_img = Image.open(img_path).convert('RGB')
      data = self.transforms(pil_img)
      return data

    def __len__(self):
      return len(self.imgs)
