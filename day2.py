import os
import cv2
import numpy as np 
from config import INPUT_DIR , OUTPUT_DIR

def imread_unicode(path):
    with open(path, "rb") as f:
        data = f.read()
    img = np.frombuffer(data, np.uint8)                         #binary data → NumPy array    (unidentify int 8)
    return cv2.imdecode(img, cv2.IMREAD_COLOR)        #NumPy array → ภาพ OpenCV         IMREAD_COLOR = อ่านภาพสี


# ---------- โหลด Haar Cascade ----------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



def detect_and_crop_faces():
    person_counter = 1  # person_1, person_2, ...

    for filename in os.listdir(INPUT_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(INPUT_DIR, filename)
        img = imread_unicode(img_path)

        if img is None:
            print(f"⚠️ อ่านรูปไม่ได้: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(80, 80)
        )

        print(f"📸 {filename} → พบ {len(faces)} หน้า")

        for i, (x, y, w, h) in enumerate(faces):
            face_crop = img[y:y+h, x:x+w]

            person_folder = os.path.join(
                OUTPUT_DIR, f"person_{person_counter}"
            )
            os.makedirs(person_folder, exist_ok=True)

            save_path = os.path.join(
                person_folder, f"{os.path.splitext(filename)[0]}_{i}.jpg"
            )

            cv2.imwrite(save_path, face_crop)
            print(f"   ✅ บันทึก: {save_path}")

            person_counter += 1



if __name__ == "__main__":
    detect_and_crop_faces()
    print("🎉 Day 2 เสร็จสมบูรณ์")