import torch
from ultralytics import YOLO


if __name__ == '__main__':
    # Ensure GPU is used if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load YOLO model
    model = YOLO("C:/CarlaYolo/runs/detect/train30/weights/best.pt")  

    # Train the model with GPU, smaller batch size for 6GB VRAM, and AMP disabled
    model.train(
        data="C:/CarlaYolo/dataset_CARLA/data.yaml",
        epochs=7,
        imgsz=640,
        device=device,  # Explicitly set device
        batch=8,        # Reduce batch size for better VRAM handling
        amp=False       # Disable AMP due to compatibility issues
    )
    


    #model.predict(source="C:/CarlaYolo/dataset_CARLA/images/val", save=True)


