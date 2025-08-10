import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_PATH = "./tomato_special.pth"  # Change as needed
TEST_FOLDER = "./test"

# Image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model_and_metadata(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(num_features, num_classes)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, class_names

def predict_single_image(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
        return class_names[predicted.item()]

def format_disease_name(class_name):
    if '___' in class_name:
        return class_name.split('___')[1].replace('_', ' ').title()
    elif '-' in class_name:
        return class_name.split('-')[1].strip().title()
    else:
        return class_name.replace('_', ' ').title()

def main():
    model, class_names = load_model_and_metadata(MODEL_PATH)
    
    image_files = [f for f in os.listdir(TEST_FOLDER) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_name in sorted(image_files):
        predicted_class = predict_single_image(model, os.path.join(TEST_FOLDER, image_name), class_names)
        print(format_disease_name(predicted_class))

if __name__ == "__main__":
    main()
