import torch
from ultralytics import YOLO

def main():
    # Load a model
    model = YOLO("models\yolo11n.pt")
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Train the model on the dataset
    model.train(
        data="datasets\yolo_CropOrWeed2\data.yaml",
        batch = -1,
        epochs=300, 
        imgsz=224,
        lr0=0.01,
        lrf=0.2,
        device=device,
        pretrained=True,
        patience=20,
        momentum=0.9,
        optimizer="Adam")
    
    # Save the trained model
    model_path = "models\\trained_yolov11n.pt"
    model.save(model_path)
    print(f"Model saved to {model_path}\n")
    
    # Validate the model on the validation set
    metrics = model.val()
    print("Validation Metrics:", metrics)
    
    # Run inference on a test image
    test_image = "C:\\Users\\yurim\\Documents\\University\\UM\\Year_3\\Bachelor_Thesis\\Weed_Detection_eSys\\datasets\\data\\images\\ave-0048-0004.jpg" 
    results = model.predict(source=test_image)
    if results: 
        results[0].show()
    else:
        print("No results to display.")

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
