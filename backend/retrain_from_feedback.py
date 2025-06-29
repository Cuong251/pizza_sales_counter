import os
import shutil
from ultralytics import YOLO

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
FEEDBACK_IMG_DIR = os.path.join(DATASET_DIR, "feedback_candidates")
FEEDBACK_LABEL_DIR = os.path.join(DATASET_DIR, "feedback_labels")
CUSTOM_DATASET = os.path.join(DATASET_DIR, "custom_dataset")
TRAIN_DIR = os.path.join(CUSTOM_DATASET, "images", "train")
LABEL_TRAIN_DIR = os.path.join(CUSTOM_DATASET, "labels", "train")

# === Setup dirs ===
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(LABEL_TRAIN_DIR, exist_ok=True)

# === Copy feedback data into custom dataset ===
print("[INFO] Preparing dataset from feedback...")
count = 0
for filename in os.listdir(FEEDBACK_IMG_DIR):
    if not filename.lower().endswith((".jpg", ".png")):
        continue

    name_no_ext = os.path.splitext(filename)[0]
    label_file = name_no_ext + ".txt"

    img_src = os.path.join(FEEDBACK_IMG_DIR, filename)
    label_src = os.path.join(FEEDBACK_LABEL_DIR, label_file)

    if not os.path.exists(label_src):
        continue  # Skip if no annotation

    img_dst = os.path.join(TRAIN_DIR, filename)
    label_dst = os.path.join(LABEL_TRAIN_DIR, label_file)

    shutil.copy(img_src, img_dst)
    shutil.copy(label_src, label_dst)
    count += 1

print(f"[INFO] Prepared {count} image-label pairs.")

# === Create YAML config for training ===
yaml_path = os.path.join(CUSTOM_DATASET, "pizza.yaml")
with open(yaml_path, "w") as f:
    f.write(f"""\
path: {CUSTOM_DATASET}
train: images/train
val: images/train
names: [pizza]
""")

# === Retrain YOLO model ===
print("[INFO] Starting YOLO training...")
model = YOLO("yolov8n.pt")  # or yolov8s.pt if you prefer
model.train(data=yaml_path, epochs=20, imgsz=640, project="retrained", name="pizza_model")

print("[✅] Training completed.")
