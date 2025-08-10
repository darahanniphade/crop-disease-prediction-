
# 🌾 Crop Disease Detection ML Model with Flask Integration

models are on drive - https://drive.google.com/drive/folders/1NbRmcgxxcOpYHrwD7LfpjZKaPq9JKDa5?usp=sharing

## 📌 Overview
This project is a **deep learning–based crop disease detection system** trained on **55,000+ labeled images** of:

**Tomato | Rice | Corn | Sugarcane | Guava | Cotton | Mango**

The model can classify multiple diseases for each crop and provide **instant predictions** via a **Flask-based web application**.

---

## 🚀 Features
- **High-Accuracy Predictions** – Detects crop diseases with confidence scores.
- **Real-World Tested** – Validated on Google-sourced images for robustness.
- **GPU-Trained Model** – Leveraged GPU acceleration for faster training.
- **Transfer Learning** – Used MobileNetV2 for efficient and accurate inference.
- **User-Friendly Web App** – Upload an image and get results instantly.
- **Data Augmentation** – Improved performance on unseen images.
- **Scalable Architecture** – Easy to extend for more crops/diseases.

---

## 🖼 Example Crops & Diseases
- **Tomato:** Early Blight, Late Blight, Leaf Curl, Mosaic Virus, Healthy  
- **Rice:** Brown Spot, Leaf Blast, Sheath Blight, Healthy  
- **Corn:** Common Rust, Northern Leaf Blight, Gray Leaf Spot, Healthy  
- **Sugarcane:** Red Rot, Smut, Healthy  
- **Guava:** Anthracnose, Wilt, Healthy  
- **Cotton:** Leaf Spot, Boll Rot, Healthy  
- **Mango:** Anthracnose, Bacterial Canker, Healthy  

---

## 🛠 Technologies Used
- **Frameworks & Libraries:** PyTorch, Flask, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Pillow
- **Model Architecture:** MobileNetV2 (Transfer Learning)
- **Hardware:** Trained on GPU (CUDA-enabled)
- **Backend:** Flask REST API
- **Frontend:** HTML, CSS, JavaScript

---

## 📂 Project Structure
├── data/ # Dataset (train/test)
├── notebooks/ # Jupyter notebooks for EDA & experiments
├── models/ # Saved PyTorch model files
├── static/ # Frontend assets (CSS, JS)
├── templates/ # HTML templates for Flask
├── app.py # Flask application
├── predict.py # Script for single image prediction
├── requirements.txt # Python dependencies
└── README.md # Project documentation


---

## 📊 Dataset
- **Total Images:** 55,000+
- **Source:** Combination of open datasets & curated Google images
- **Annotations:** Crop type + disease label
- **Preprocessing:** Resizing, normalization, augmentation (rotation, brightness, noise, crop, cutmix)

---

## ⚡ Model Training
1. **Architecture:** MobileNetV2 pretrained on ImageNet
2. **Optimizer:** Adam (lr = 0.001)
3. **Loss Function:** CrossEntropyLoss
4. **Batch Size:** 32
5. **Epochs:** 25–50 (early stopping)
6. **Data Augmentation:**  
   - Random rotation, flipping, brightness/contrast shift  
   - Gaussian noise, random cropping, CutMix
7. **Training Hardware:** NVIDIA GPU with CUDA support

---

## 🌐 Flask Web App
The Flask app allows users to upload a crop image and get:
- Predicted disease label
- Confidence score
- (Optional) Heatmap showing affected area

**Endpoints:**
- POST /predict → Returns prediction JSON
- GET / → Web UI for uploading images

---

## ▶️ How to Run Locally
### 1️⃣ Clone the repository

git clone https://github.com/yourusername/crop-disease-detection.git
cd crop-disease-detection

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Flask app
python app.py
