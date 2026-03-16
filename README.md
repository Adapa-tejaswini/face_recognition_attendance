# Face Recognition Attendance System

This project is a Python-based Face Recognition Attendance System that detects faces using a webcam and automatically records attendance.

## Features
- Face detection using OpenCV
- Face recognition using face_recognition library
- Automatic attendance marking
- Attendance stored in CSV file

## Project Structure

face_attandence/
│
├── dataset_face/
├── attendance.csv
├── encode_faces.py
├── recognize_faces.py
├── encodings.pickle
└── README.md

## Technologies Used

## Technologies Used

| Technology | Purpose |
|-----------|--------|
| Python | Core programming language used to build the system |
| OpenCV | Used for image processing and webcam access |
| face_recognition | Library used for detecting and recognizing faces |
| NumPy | Used for numerical operations on face encoding data |
| Pickle | Used to store and load face encodings efficiently |
| CSV | Used to store attendance records |

## How to Run

### 1 Install dependencies

pip install opencv-python face-recognition numpy

### 2 Encode faces

python encode_faces.py

### 3 Start recognition

python recognize_faces.py

## Output

The system detects faces and records attendance in **attendance.csv**.