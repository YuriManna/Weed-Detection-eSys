import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class for defining image and label loading, applying transformations
class CustomDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Image Path and image loading
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Label Path and label loading
        label_path = os.path.join(self.annotations_dir, img_name.replace('.jpg', '.txt'))
        with open(label_path, 'r') as f:
            bboxes = [list(map(float, line.strip().split())) for line in f.readlines()] 
        class_labels = [bbox[0] for bbox in bboxes]
        boxes = [bbox[1:] for bbox in bboxes]

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, class_labels=class_labels)
            image = augmented['image']
            boxes = augmented['bboxes'] #TODO: might need to convert to numpy array
            class_labels = augmented['class_labels']

        # Convert image and labels to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Convert to tensor, channel-first, normalize to [0, 1]
        labels = torch.tensor([[cls] + box for cls, box in zip(class_labels, boxes)], dtype=torch.float32)

        return  image, labels
    
    @staticmethod
    def collate_fn(batch):
        images = []
        labels = []

        for image, label in batch:
            images.append(image)
            labels.append(label)

        images = torch.stack(images, dim=0)
        labels = torch.cat(labels, dim=0)  

        return images, labels