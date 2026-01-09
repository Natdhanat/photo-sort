import os
import cv2
import numpy as np

from config import OUTPUT_DAY2_DIR, OUTPUT_DAY3_DIR, MODEL_PATH

print("🚀 Day 3: Face Embedding & Grouping")
print("=" * 50)

# ==================================================
# 1) โหลดโมเดล OpenFace
# ==================================================
print("📦 Loading OpenFace model...")
net = cv2.dnn.readNetFromTorch(MODEL_PATH)
print("✅ Model loaded successfully\n")


# ==================================================
# 2) แปลงภาพใบหน้า → embedding vector
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
# 4) จัดกลุ่มใบหน้า
# ==================================================
def group_faces():
    print("📂 Reading faces from Day 2 output...")
    os.makedirs(OUTPUT_DAY3_DIR, exist_ok=True)

    known_embeddings = []
    person_folders = []
    person_count = 1
    total_faces = 0

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

            for i, known_emb in enumerate(known_embeddings):
                dist = euclidean_distance(embedding, known_emb)
                print(f"   📏 Distance to person_{i+1}: {dist:.4f}")

                if dist < 0.9:
                    save_dir = person_folders[i]
                    cv2.imwrite(os.path.join(save_dir, filename), face_img)
                    print(f"   ✅ Matched → saved to {os.path.basename(save_dir)}")
                    matched = True
                    break

            if not matched:
                person_dir = os.path.join(
                    OUTPUT_DAY3_DIR,
                    f"person_{person_count}"
                )
                os.makedirs(person_dir, exist_ok=True)

                cv2.imwrite(os.path.join(person_dir, filename), face_img)

                known_embeddings.append(embedding)
                person_folders.append(person_dir)

                print(f"   🆕 New person detected → person_{person_count}")
                person_count += 1

    print("\n🎉 Day 3 completed")
    print(f"👥 Total persons: {person_count - 1}")
    print(f"📸 Total faces processed: {total_faces}")


# ==================================================
# 5) เรียกใช้งานจริง (ไม่ใช่แค่ define)
# ==================================================
if __name__ == "__main__":
    print("▶️ Starting face grouping...\n")
    group_faces()
    print("\n✅ Program finished successfully")
