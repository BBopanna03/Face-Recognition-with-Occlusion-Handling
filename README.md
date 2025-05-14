# Face-Recognition-with-Occlusion-Handling
The project implements a robust face recognition system capable of handling occlusions like masks and disguises. It includes utilities to verify image data quality and trains a recognition model using HOG or CNN-based encoders. The system is optimized for both CPU and GPU, ensuring flexible deployment.

---

## 📁 File Structure

```plaintext
YOLOv8/
├── data/                              # Data directory for face training images
│   └── faces/                         # Main face data folder
│       ├── Person_1/                  # Subfolders for each person
│       │   ├── img1.jpg
│       │   ├── img2.png
│       │   └── ...
│       ├── Person_2/
│       └── ...
├── models/
│   ├── occlusion_model.pkl            # Occlusion classification model (mask/disguise)
│   └── yolov8s.pt                     # YOLOv8 pre-trained model for face detection
├── utils/ 
│   ├── __init__.py                    # (optional) to make it a package 
│   ├── face_encoder.py                # Encodes face images using HOG or CNN
│   └── face_matcher.py                # Trains, saves, and loads face recognition model                  
├── advanced_occlusion_recognition.py  # Main application for training and recognition
├── check_data.py                      # Script to check the validity of data
├── check_images.py                    # Script to validate images with face detection
├── collect_faces.py                   # Script to collect face images from webcam
├── download_model.py                  # Script to download required models
├── recognize_faces.py                 # Script for face recognition application
├── requirements.txt                   # File to install dependencies
├── simple_collect.py                  # Script to collect face images in a simpler way
├── simple_face_rec.py                 # Simple face recognition script
└── train_recognizer.py                # Script to train the face recognition model


---


## 📥 Model Downloads

Download the required models from the following Google Drive link and place them in the project root:

👉 [https://drive.google.com/drive/folders/1B7SRy73KPMyXnpEqCDyZJwEpZx_VVVhY?usp=drive_link](#) 

---

## 🚀 How to Run the Application

### 📸 1. Collect Face Images

Use this command to collect face images using your webcam:

    ```bash
    python simple_collect.py --name "Your_Name" --num-images 30
🔁 The more images you collect per person, the better the recognition accuracy.

🧠 2. Run the Application
To start the occlusion-aware recognition system, run:

    python advanced_occlusion_recognition.py
    You’ll be prompted with:

    Occlusion-Resistant Face Recognition
        1. Train model 
        2. Run recognition
        Enter choice (1/2): 
        Press 1 to train the model using the collected images.
        Press 2 to start the real-time recognition and occlusion detection application.

✅ Requirements
Python 3.8+
ultralytics 8.0.0
opencv-python 4.7.0
numpy 1.23.5
scikit-learn 1.2.2
face-recognition 1.3.0
tqdm 4.65.0

Install dependencies with:

    pip install -r requirements.txt

📌 Notes
->Make sure your webcam is enabled and accessible.
->For accurate recognition, collect images in different lighting and angles.