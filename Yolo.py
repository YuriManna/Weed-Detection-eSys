from ultralytics import YOLO
import torch
import albumentations as A


def train_model(weights, data, img_size, epochs, batch_size, device, name):
    """
    Train a YOLO model.
    
    Args:
        weights (str): Path to pre-trained weights or 'yolov11s.pt' (or other variants).
        data (str): Path to dataset.yaml.
        img_size (int): Image size for training.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        device (torch.device): Device to run the training on (CPU or CUDA).
    """
    # Load the YOLO model
    model = YOLO(weights).to(device)
    
    
    # Train the model
    model.train(
        data=data,          
        imgsz=img_size,     
        epochs=epochs,     
        batch=batch_size,   
        name=name,
    )
    print("Training completed. Check 'runs/train/weed_detection' for results.")

def test_model(weights, data, img_size, conf_thresh, iou_thresh, device):
    """
    Test a YOLO model on the test dataset.
    
    Args:
        weights (str): Path to the trained weights file (e.g., best.pt).
        data (str): Path to dataset.yaml.   
        img_size (int): Image size for inference.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IOU threshold for NMS.
    """
    # Load the model
    model = YOLO(weights).to(device)
    
    # Run the test
    results = model.val(
        data=data,           
        imgsz=img_size,     
        conf=conf_thresh,   
        iou=iou_thresh,     
        split="test",
    )


def print_logo():
    logo = """
  ___    ___ ________  ___       ________     
 |\  \  /  /|\   __  \|\  \     |\   __  \    
 \ \  \/  / | \  \|\  \ \  \    \ \  \|\  \   
  \ \    / / \ \  \\\   \ \  \    \ \  \\\   \  
   \/  /  /   \ \  \\\   \ \  \____\ \  \\\   \ 
 __/  / /      \ \_______\ \_______\ \_______\\
|\___/ /        \|_______|\|_______|\|_______|
\|___|/                                       
            """
    print(logo)

if __name__ == "__main__":
    print_logo()

    # Get the mode from the user
    mode = input("Please enter the mode to run (train, test): ").strip().lower()
    # Validate the input
    while mode not in ["train", "test"]:
        if mode == "exit":
            print("Exiting the program.")
            exit()
        print("Invalid mode. Please choose from 'train' or 'test'.")
        mode = input("Please enter the mode to run (train, test): ").strip().lower()

    # Common parameters
    weights = "yolo\yolov11n.pt"  # Path to weights or pre-trained model
    data = "WeedCrop\data.yaml"  # Path to dataset configuration   TODO adapt to the dataset
    img_size = 416  # Image size for training and testing

    # Get the model name if training    
    if mode == "train":
        name = input("Please enter the name of the model: ").strip().lower()

    # ask for which model to use for validation and testing
    elif mode == "test":
        weights = input("Please enter the path to the weights file: ").strip().lower()

    print(f"Running in {mode} mode...\n")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if mode == "train":     
        epochs = 70
        batch_size = 64
        train_model(weights, data, img_size, epochs, batch_size, device, name, )
    elif mode == "test":
        conf_thresh = 0.25
        iou_thresh = 0.45
        test_model(weights, data, img_size, conf_thresh, iou_thresh, device)