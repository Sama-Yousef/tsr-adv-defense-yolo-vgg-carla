import torch
import cv2
import os
from ultralytics import YOLO
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the classifier model (VGG16)
classifier_model = models.vgg16(pretrained=False)
classifier_model.classifier[6] = torch.nn.Linear(4096, 43)  # Adjust output classes
classifier_model.load_state_dict(torch.load("C:/CarlaYolo/attacks/vgg_clean.pth", map_location=device))
classifier_model.to(device)
classifier_model.eval()  # Set to evaluation mode

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Mapping from class index to sign label
class_map = {
    0: "Speed Limit 20",
    1: "Speed Limit 30",
    2: "Speed Limit 50",
    3: "Speed Limit 60",
    4: "Speed Limit 70",
    5: "Speed Limit 80",
    6: "End of Speed Limit 80",
    7: "Speed Limit 100",
    8: "Speed Limit 120",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons"
}

# Function to predict class of a cropped image
def predict_image(cropped_img):
    image = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  # Convert to PIL Image
    image = transform(image).unsqueeze(0).to(device)  # Apply transformations

    with torch.no_grad():
        outputs = classifier_model(image)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()

# Load the trained YOLO model
yolo_model_path = "C:/CarlaYolo/runs/detect/train30/weights/best.pt"
yolo_model = YOLO(yolo_model_path)

# Define paths
raw_images_folder = "C:/CarlaYolo/CudaPytorch/Scripts/outputSTOP1Town10HD_Opt__WetFalse__CloudFalse__RainLvl3Stop_Sign"
predict_folder = "predict_stop_clean"
os.makedirs(predict_folder, exist_ok=True)

# Process each image
for image_file in os.listdir(raw_images_folder):
    if not image_file.endswith((".jpg", ".png", ".jpeg")):
        continue

    # Load the image
    image_path = os.path.join(raw_images_folder, image_file)
    image = cv2.imread(image_path)

    # Run YOLO inference
    results = yolo_model(image_path)

    # Loop through detected objects
    for i, result in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, result[:4])  # Bounding box coordinates

        # Crop the image
        cropped_image = image[y1:y2, x1:x2]

        # Ensure valid crop
        if cropped_image.size == 0:
            continue

        # Get the predicted label index
        predicted_label = predict_image(cropped_image)

        # Map to sign label
        label_text = class_map.get(predicted_label, str(predicted_label))

        # Draw bounding box and label on the original image
        color = (0, 255, 0)
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(image, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    # Save the annotated image
    save_path = os.path.join(predict_folder, image_file)
    cv2.imwrite(save_path, image)

print("Processing completed! Check the 'predict' folder for results.")
