# Face Recognition using OpenCV

## Overview

This project is a real-time Face Recognition System developed using Python and OpenCV. It detects and recognizes human faces from a live webcam feed using the Haar Cascade Classifier for face detection and the Local Binary Pattern Histogram (LBPH) Face Recognizer for identification.

The project demonstrates practical applications of computer vision, image processing, and machine learning for identity recognition.

---

## Features

* Real-time face detection using webcam
* Face recognition using the LBPH algorithm
* Face dataset generation
* Model training for recognized users
* Live recognition with confidence score
* Easy to extend for attendance and security systems

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Haar Cascade Classifier
* LBPH Face Recognizer

---

## Project Structure

```
Face-Recognition/
│
├── capture.py
├── train.py
├── recognize.py
├── haarcascade_frontalface_default.xml
├── requirements.txt
├── dataset/
├── trainer/
├── images/
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/Amith786/Face-Recognition.git
cd Face-Recognition
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1: Capture Face Dataset

```bash
python capture.py
```

### Step 2: Train the Model

```bash
python train.py
```

### Step 3: Start Face Recognition

```bash
python recognize.py
```

---

## Applications

* Smart Attendance System
* Access Control
* Identity Verification
* Security and Surveillance
* Computer Vision Research

---

## Future Enhancements

* Deep Learning-based Face Recognition
* Face Mask Detection
* Face Anti-Spoofing
* Emotion Recognition
* Cloud Database Integration

---

## Author

**Amith Anand**

AI & Data Science Graduate

GitHub: https://github.com/Amith786
