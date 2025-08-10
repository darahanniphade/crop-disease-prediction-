from flask import Flask, render_template, request
import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model paths
MODEL_PATHS = {
    "tomato": "./tomato_special.pth",
    "other": "./all.pth"
}

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(model_path):
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

def predict_image(model, image_path, class_names):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        _, predicted = torch.max(probabilities, 1)
        return clean_name(class_names[predicted.item()])

def clean_name(class_name):
    if '___' in class_name:
        return class_name.split('___')[1].replace('_', ' ').title()
    elif '-' in class_name:
        return class_name.split('-')[1].strip().title()
    else:
        return class_name.replace('_', ' ').title()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        crop_type = request.form.get("crop_type")
        file = request.files["image"]

        if file and crop_type in MODEL_PATHS:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            model, class_names = load_model(MODEL_PATHS[crop_type])
            prediction = predict_image(model, filepath, class_names)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    app.run(debug=True)
