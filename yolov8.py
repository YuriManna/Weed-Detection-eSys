import torch
from ultralytics import YOLO

def main():
    # Load a pretrained YOLOv8 small model
    model = YOLO("yolov8m.pt")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Train the model on your dataset
    # Ensure you have a "data.yaml" file that specifies your dataset paths and class names
    model.train(data="datasets\yolo_CropOrWeed2\data.yaml", epochs=50, imgsz=640)
    
    # Save the trained model
    model_path = "trained_yolov8m.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}\n")
    
    # Validate the model on the validation set
    metrics = model.val()
    print("Validation Metrics:", metrics)
    
    # Run inference on a test image
    test_image = "C:\\Users\\yurim\\Documents\\University\\UM\\Year_3\\Bachelor_Thesis\\Weed_Detection_eSys\\datasets\\data\\images\\ave-0048-0004.jpg" 
    results = model.predict(source=test_image)
    results.show()

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
    main()
