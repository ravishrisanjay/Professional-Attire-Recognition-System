# 🎯 AI Professional Attire Recognition System

A Python-based AI system that automatically checks whether students are adhering to a formal dress code by analyzing live webcam input, photos, or video files using **YOLOv8** and **OpenCV**.

---

## 👀 What It Detects

- ✅ Wearing an **ID card**
- ✅ Wearing **shoes**
- ✅ **Shirt is tucked in** (formal belt detected)

### 🚦 Output Behavior

- 🟩 If **all three** conditions are met, the student is highlighted with a **green bounding box**.
- 🟥 If **any** requirement is missing, the student is marked with a **red bounding box**.

---

## 🛠 Technologies Used

- **Python 3.8+**
- **YOLOv8** (via [Ultralytics](https://github.com/ultralytics/ultralytics))
- **OpenCV** – for real-time image and video processing

---

## 📁 Download Pre-trained Model Files (.pt)

Due to size restrictions, the YOLOv8 model and custom-trained weights are not included in the GitHub repository.

📥 **Download all `.pt` model files from this Google Drive folder**:

👉 [Download .pt files from Google Drive](https://drive.google.com/drive/folders/13oLS_1nBvpvE6ZT-zb1WU7TFaJptplqF?usp=sharing)

After downloading, **place all `.pt` files in the root folder** of the project:

/Professional-Attire-Recognition-System/
├── id_card.pt
├── belt.pt
├── shoes.pt

📥 Install Required Packages
bash
pip install ultralytics opencv-python

RUN - App.py
