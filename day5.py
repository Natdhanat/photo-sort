import os
import cv2
import numpy as np

from config import OUTPUT_DAY2_DIR, OUTPUT_DAY5_DIR, MODEL_PATH

print("🚀 Day 5: Face Grouping with Average Embedding (Centroid)")
print("=" * 60)

# ==================================================
# 1) โหลด OpenFace model
# ==================================================
print("📦 Loading OpenFace model...")
net = cv2.dnn.readNetFromTorch(MODEL_PATH)
print("✅ Model loaded successfully\n")


# ==================================================
# 2) แปลงภาพใบหน้าเป็น embedding
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
# 3) Euclidean distance
# ==================================================
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


# ==================================================
# 4) คำนวณค่าเฉลี่ย embedding ของคนหนึ่งคน
# ==================================================
def compute_centroid(embeddings):
    return np.mean(embeddings, axis=0)


# ==================================================
# 5) Grouping แบบใช้ centroid
# ==================================================
def group_faces():
    print("📂 Reading faces from Day 2 output...")
    os.makedirs(OUTPUT_DAY5_DIR, exist_ok=True)

    persons = []  
    # structure:
    # persons = [
    #   {
    #       "embeddings": [...],
    #       "centroid": np.array,
    #       "dir": "path"
    #   }
    # ]

    person_count = 1
    total_faces = 0
    THRESHOLD = 0.65   # ปรับให้เข้มขึ้น

    for folder in os.listdir(OUTPUT_DAY2_DIR):
        folder_path = os.path.join(OUTPUT_DAY2_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        print(f"\n➡️ Processing folder: {folder}")

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder_path, filename)
            face_img = cv2.imread(img_path)

            if face_img is None:
                print(f"⚠️ Cannot read image: {img_path}")
                continue

            total_faces += 1
            print(f"👤 Face #{total_faces}: {filename}")

            embedding = get_face_embedding(face_img)
            print("   🔢 Embedding created")

            matched = False

            for i, person in enumerate(persons):
                dist = euclidean_distance(embedding, person["centroid"])
                print(f"   📏 Distance to person_{i+1} centroid: {dist:.4f}")

                if dist < THRESHOLD:
                    cv2.imwrite(
                        os.path.join(person["dir"], filename),
                        face_img
                    )

                    person["embeddings"].append(embedding)
                    person["centroid"] = compute_centroid(
                        person["embeddings"]
                    )

                    print(f"   ✅ Matched → person_{i+1}")
                    matched = True
                    break

            if not matched:
                person_dir = os.path.join(
                    OUTPUT_DAY5_DIR,
                    f"person_{person_count}"
                )
                os.makedirs(person_dir, exist_ok=True)

                cv2.imwrite(
                    os.path.join(person_dir, filename),
                    face_img
                )

                persons.append({
                    "embeddings": [embedding],
                    "centroid": embedding,
                    "dir": person_dir
                })

                print(f"   🆕 New person detected → person_{person_count}")
                person_count += 1

    print("\n🎉 Day 5 completed")
    print(f"👥 Total persons: {person_count - 1}")
    print(f"📸 Total faces processed: {total_faces}")


# ==================================================
# 6) Run
# ==================================================
if __name__ == "__main__":
    print("▶️ Starting improved face grouping...\n")
    group_faces()
    print("\n✅ Program finished successfully")
