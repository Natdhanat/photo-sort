import os
import cv2
import numpy as np

from config import OUTPUT_DAY5_DIR, MODEL_PATH

print("📊 Day 6: Face Group Evaluation")
print("=" * 60)

# ==================================================
# 1) Load OpenFace model
# ==================================================
net = cv2.dnn.readNetFromTorch(MODEL_PATH)


# ==================================================
# 2) Create embedding
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
def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


# ==================================================
# 4) Analyze each person folder
# ==================================================
def analyze_groups():
    person_embeddings = {}

    # -----------------------------
    # Load embeddings per person
    # -----------------------------
    for person in os.listdir(OUTPUT_DAY5_DIR):
        person_path = os.path.join(OUTPUT_DAY5_DIR, person)

        if not os.path.isdir(person_path):
            continue

        embeddings = []

        for file in os.listdir(person_path):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            emb = get_face_embedding(img)
            embeddings.append((file, emb))

        if len(embeddings) >= 2:
            person_embeddings[person] = embeddings

    # -----------------------------
    # Intra-person distance
    # -----------------------------
    print("\n🔍 Intra-person distance analysis")
    print("-" * 60)

    for person, emb_list in person_embeddings.items():
        print(f"\n👤 Analyzing {person}")
        print("-" * 50)

        for i in range(len(emb_list)):
            for j in range(i + 1, len(emb_list)):
                f1, e1 = emb_list[i]
                f2, e2 = emb_list[j]

                dist = euclidean_distance(e1, e2)
                print(f"📏 {f1} ↔ {f2} => distance = {dist:.4f}")

    # -----------------------------
    # Inter-person distance
    # -----------------------------
    print("\n🔁 Inter-person distance (representative)")
    print("-" * 60)

    persons = list(person_embeddings.keys())

    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1 = persons[i]
            p2 = persons[j]

            e1 = person_embeddings[p1][0][1]
            e2 = person_embeddings[p2][0][1]

            dist = euclidean_distance(e1, e2)
            print(f"👥 {p1} ↔ {p2} => distance = {dist:.4f}")

    print("\n✅ Day 6 evaluation completed")


# ==================================================
# 5) Run
# ==================================================
if __name__ == "__main__":
    analyze_groups()
