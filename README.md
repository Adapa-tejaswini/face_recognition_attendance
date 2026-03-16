# 🎓 Face Recognition Attendance System

A Python-based system that detects faces using a webcam and automatically records attendance.

---

## 🚀 Features

✔️ Face detection using webcam  
✔️ Face recognition using trained encodings  
✔️ Automatic attendance marking  
✔️ Attendance stored in CSV file  

---

## 🛠️ Technologies Used

| Technology | Category | Purpose |
|-----------|----------|--------|
| Python | Programming Language | Core language used to develop the system |
| OpenCV | Computer Vision Library | Handles image processing and webcam input |
| face_recognition | Machine Learning Library | Performs face detection and recognition |
| NumPy | Numerical Library | Handles face encoding vector operations |
| Pickle | Serialization Library | Saves and loads encoded face data |
| CSV | Data Storage | Stores attendance records |

---

## ⚙️ How to Run

### 1️⃣ Install dependencies

pip install opencv-python face-recognition numpy

### 2️⃣ Encode faces

python encode_faces.py

### 3️⃣ Start recognition

python recognize_faces.py

---

## 📊 Output

The system detects faces and records attendance automatically in **attendance.csv**.

