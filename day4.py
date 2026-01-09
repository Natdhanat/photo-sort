import os
import cv2
import numpy as np

from config import OUTPUT_DAY3_DIR, MODEL_PATH

print("📊 Day 4: Face Similarity Analysis")
print("=" * 60)

# ==================================================
# 1) โหลดโมเดล OpenFace
# ==================================================
print("📦 Loading OpenFace model...")
net = cv2.dnn.readNetFromTorch(MODEL_PATH)
print("✅ Model loaded\n")


# ==================================================
# 2) สร้าง embedding จากภาพใบหน้า
# ==================================================
def get_face_embedding(face_img):
    face_img = cv2.resize(face_img, (96, 96))

    blob = cv2.dnn.blobFromImage(
        face_img,
        scalefactor=1.0 / 255,
        size=(96, 96),
        mean=(0, 0, 0),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)
    embedding = net.forward()

    return embedding.flatten()


# ==================================================
# 3) คำนวณ Euclidean Distance
# ==================================================
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# ==================================================
# 4) วิเคราะห์ความเหมือนในแต่ละ person_x
# ==================================================
def analyze_similarity():
    print("📂 Reading grouped faces from Day 3...\n")

    for person_folder in sorted(os.listdir(OUTPUT_DAY3_DIR)):
        person_path = os.path.join(OUTPUT_DAY3_DIR, person_folder)

        if not os.path.isdir(person_path):
            continue

        print(f"👤 Analyzing {person_folder}")
        print("-" * 50)

        embeddings = []
        filenames = []

        for filename in os.listdir(person_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Cannot read image: {filename}")
                continue

            emb = get_face_embedding(img)
            embeddings.append(emb)
            filenames.append(filename)

        # เปรียบเทียบทุกคู่
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                dist = euclidean_distance(
                    embeddings[i],
                    embeddings[j]
                )
                print(
                    f"📏 {filenames[i]} ↔ {filenames[j]} "
                    f"=> distance = {dist:.4f}"
                )

        if len(embeddings) < 2:
            print("ℹ️ Not enough images to compare")

        print()


# ==================================================
# 5) เรียกใช้งานจริง
# ==================================================
if __name__ == "__main__":
    print("▶️ Starting similarity analysis...\n")
    analyze_similarity()
    print("✅ Day 4 finished successfully")
