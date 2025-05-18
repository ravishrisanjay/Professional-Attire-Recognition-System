import cv2 as cv
from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time

# Load models
id_model = YOLO('H:/mini project/Project/id_card_dataset.pt')
belt_model = YOLO('H:/mini project/Project/belt_dataset.pt')
shoe_model = YOLO('H:/mini project/Project/shoes_dataset.pt')
person_model = YOLO('H:/mini project/Project/yolov8x.pt')

save_dir = "H:/mini project/Project/results"
os.makedirs(save_dir, exist_ok=True)

# === Logic ===
def process_image(image_path):
    temp_img = cv.imread(image_path)
    if temp_img is None:
        messagebox.showerror("Error", "Could not read image!")
        return None

    output_img = temp_img.copy()
    persons = person_model.predict(temp_img, classes=0, conf=0.5, save=False)

    for p in persons:
        for box in p.boxes:
            b = box.xyxy[0]
            cropped_person = temp_img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            ids = id_model.predict(cropped_person, save=False)
            belts = belt_model.predict(cropped_person, save=False)
            shoes = shoe_model.predict(cropped_person, save=False)

            color = (0, 255, 0) if (len(ids[0].boxes.cls) >= 1 and len(belts[0].boxes.cls) >= 1 and len(shoes[0].boxes.cls) >= 1) else (0, 0, 255)
            cv.rectangle(output_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)

    output_path = os.path.join(save_dir, "final_output.jpg")
    cv.imwrite(output_path, output_img)
    return output_path

def run_processing(filepath):
    result_label.config(text="⏳ Processing image...")
    time.sleep(0.5)
    result_path = process_image(filepath)
    if result_path:
        img = Image.open(result_path)
        img = img.resize((550, 400))
        img_tk = ImageTk.PhotoImage(img)
        image_panel.config(image=img_tk)
        image_panel.image = img_tk
        result_label.config(text="✅ Detection Completed!")

def select_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not filepath:
        return
    threading.Thread(target=run_processing, args=(filepath,), daemon=True).start()

def start_webcam():
    cap = cv.VideoCapture(0)
    i = 5
    result_label.config(text="Webcam Started – Press Q to Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i == 5:
            i = 0
            persons = person_model.predict(frame, classes=0, conf=0.5, save=False)
            for p in persons:
                for box in p.boxes:
                    b = box.xyxy[0]
                    cropped = frame[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
                    ids = id_model.predict(cropped, save=False)
                    belts = belt_model.predict(cropped, save=False)
                    shoes = shoe_model.predict(cropped, save=False)
                    color = (0, 255, 0) if (len(ids[0].boxes.cls) >= 1 and len(belts[0].boxes.cls) >= 1 and len(shoes[0].boxes.cls) >= 1) else (0, 0, 255)
                    cv.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, 3)

            cv.imshow("Live Detection – Press Q to Quit", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            i += 1

    cap.release()
    cv.destroyAllWindows()
    result_label.config(text="Webcam Detection Closed")

# === GUI Setup ===
root = tk.Tk()
root.title("Safety Compliance Detector")
root.geometry("720x650")
root.configure(bg="#f0f4f8")

# Header
header = tk.Label(root, text="🛡️ Safety Compliance Detector", font=("Helvetica", 20, "bold"), fg="#333", bg="#f0f4f8")
header.pack(pady=15)

# Action Buttons Frame
btn_frame = tk.Frame(root, bg="#f0f4f8")
btn_frame.pack(pady=10)

upload_btn = tk.Button(btn_frame, text="Upload Image", command=select_image, width=20, bg="#007acc", fg="white", font=("Helvetica", 12))
upload_btn.grid(row=0, column=0, padx=15)

webcam_btn = tk.Button(btn_frame, text="Start Webcam", command=start_webcam, width=20, bg="#007acc", fg="white", font=("Helvetica", 12))
webcam_btn.grid(row=0, column=1, padx=15)

# Result Status Label
result_label = tk.Label(root, text="Upload an image or start webcam to begin detection", font=("Helvetica", 12), fg="#555", bg="#f0f4f8")
result_label.pack(pady=5)

# Image Preview Frame
preview_frame = tk.LabelFrame(root, text="Detection Preview", font=("Helvetica", 12, "bold"), padx=10, pady=10, bg="#ffffff")
preview_frame.pack(pady=20)

image_panel = tk.Label(preview_frame, bg="white")
image_panel.pack()

root.mainloop()
