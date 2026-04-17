# 🤖 Deep Learning Based Sign Language Gesture Recognition System

## 📌 Overview

This project presents a **Deep Learning-based Sign Language Gesture Recognition System** that detects and classifies hand gestures from video input using a **3D Swin Transformer (Swin3D)** architecture.

The system processes raw video sequences, extracts spatial and temporal features, and predicts gesture classes with high accuracy. It also provides a **web-based interface** for real-time inference and visualization.

---

## 🚀 Features

* 🎥 Video-based gesture recognition
* 🧠 3D Swin Transformer model (Swin3D)
* ⚡ Real-time prediction via web interface
* 📊 Top-5 gesture predictions
* 📈 Model performance visualization (graphs)
* 📄 Auto PDF report generation
* 🎨 Premium UI (Dark theme with glassmorphism)

---

## 🧠 Model Architecture

The system uses a **3D Transformer-based model (Swin3D)** to capture both spatial and temporal features.

### Pipeline:

```
Video → Frame Extraction → Normalization → Swin3D → Classification → Output
```

### Key Techniques:

* Transfer Learning (Pretrained Swin3D)
* Mixed Precision Training (AMP)
* Cosine Learning Rate Scheduler
* Gradient Clipping
* On-the-fly Frame Extraction

---

## 📂 Project Structure

```
sign-language-app/
│
├── app.py
├── requirements.txt
│
├── model/
│   ├── best_model.pth
│   └── inference.py
│
├── utils/
│   ├── label_map.py
│   └── report_generator.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── uploads/
│   └── graphs/
│       ├── confusion_matrix.png
│       ├── metrics.png
│       └── test_accuracy.png
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd sign-language-app
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Output

The system provides:

* Predicted Gesture (e.g., Gesture A, Gesture B)
* Confidence Score
* Top-5 Predictions
* Model Performance Graphs
* Downloadable PDF Report

---

## 📸 Graphs Included

* Confusion Matrix
* Precision, Recall, F1 Score
* Test Accuracy Comparison

---

## 📄 Report Generation

The system can generate a **PDF report** including:

* Prediction results
* Confidence score
* Top-5 predictions
* Performance graphs

---

## 📚 Dataset

* **AUTSL (Ankara University Turkish Sign Language Dataset)**
* 200+ gesture classes
* Real-world variations (lighting, background, multiple users)

---

## 💻 Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Flask (Web Framework)
* Tailwind CSS (Frontend)
* ReportLab (PDF Generation)

---

## 🔮 Future Scope

* Real-time webcam integration
* Speech output (gesture → voice)
* Mobile application support
* Multilingual gesture recognition
* Deployment on cloud platforms

---

## 🧠 Conclusion

This project demonstrates an efficient and scalable approach to sign language recognition using deep learning. By leveraging a **3D Transformer architecture**, the system achieves accurate gesture classification and provides an intuitive user interface for real-world usability.

---

## 👨‍💻 Author

**Prathmesh Vilasrao Jadhav**
Bachelor of Computer Application (BCA)
Tilak Maharashtra Vidyapeeth University

---

## 📌 Note

Gesture labels are represented as generic classes (e.g., Gesture A, Gesture B) due to dataset limitations.

---

## ⭐ Acknowledgements

* AUTSL Dataset
* PyTorch Team
* Research on Vision Transformers
