# Custom Dataset class for defining image and label loading, applying transformations
class CustomDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, transform=None):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = os.listdir(img_dir)
        self.annotation_files = os.listdir(annotations_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Image Path
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Label Path and Parsing
        label_path = os.path.join(self.annotations_dir, img_name.replace('.JPG', '.txt'))
        bboxes = open_bboxes_txt(label_path)
        labels = [bbox[0] for bbox in bboxes]
        boxes = [bbox[1:] for bbox in bboxes]

        # Store original image and bounding boxes for visualization
        original_image = image.copy()
        original_bboxes_with_labels = [[labels[i]] + list(bbox) for i, bbox in enumerate(boxes)]

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, class_labels=labels)
            image = augmented['image']
            boxes = augmented['bboxes']
            labels = augmented['class_labels']
            augmented_bboxes_with_labels = [[labels[i]] + list(bbox) for i, bbox in enumerate(boxes)]
        else:
            augmented_bboxes_with_labels = original_bboxes_with_labels

        # Convert original and augmented images to tensors
        original_image = torch.from_numpy(original_image).permute(2, 0, 1).float() / 255.0  # Convert to tensor, channel-first, normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Convert to tensor, channel-first, normalize to [0, 1]

        return original_image, image, original_bboxes_with_labels, augmented_bboxes_with_labels

# Create dataset and dataloader
dataset = CustomDataset(annotations_dir='Data/luxeed_heatmaps/data/labels', img_dir='Data/luxeed_heatmaps/data/images', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
