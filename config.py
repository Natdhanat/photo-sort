import os

# ถ้าโฟลเดอร์นี้อยู่ใน Google Drive ให้แก้ BASE_DIR เป็น path จริง
# ตัวอย่าง:

BASE_DIR = r"G:\My Drive\face_project"

# ===== Google Drive paths =====

INPUT_DIR = r"G:\My Drive\face_project\input"

OUTPUT_DAY2_DIR = r"G:\My Drive\face_project\output_day2"

OUTPUT_DAY3_DIR = r"G:\My Drive\face_project\output_day3"

MODEL_PATH = r"G:\My Drive\face_project\models\openface.nn4.small2.v1.t7"


# สร้างโฟลเดอร์อัตโนมัติถ้ายังไม่มี
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DAY2_DIR, exist_ok=True)
os.makedirs(OUTPUT_DAY3_DIR, exist_ok=True)