import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import requests
import subprocess
import sys

# === Config ===
IMAGES_DIR = "../dataset/feedback_candidates"  # Folder with detection frame images
LABELS_DIR = "../dataset/feedback_labels"      # Save user-corrected labels here
CLASS_NAME = "pizza"
API_URL = "http://127.0.0.1:8000/counts"       # FastAPI endpoint

os.makedirs(LABELS_DIR, exist_ok=True)

# === Helper: Convert bounding box to YOLO format ===
def to_yolo_format(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# === Main App ===
st.set_page_config(page_title="Pizza Detection Feedback", layout="centered")
st.title("🍕 Pizza Detection Feedback")

# === Image Annotation Section ===
if not os.path.exists(IMAGES_DIR):
    st.error(f"Image folder not found: {IMAGES_DIR}")
    st.stop()

image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

if not image_files:
    st.warning("⚠️ No images found in feedback_candidates folder.")
    st.stop()

selected_file = st.selectbox("Select an image to annotate:", image_files)

if selected_file:
    img_path = os.path.join(IMAGES_DIR, selected_file)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    img_h, img_w = img_np.shape[:2]

    st.write("### 🖍️ Draw Bounding Boxes (Click + Drag)")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.3)",
        stroke_width=2,
        stroke_color="#00FF00",
        background_image=img,
        update_streamlit=True,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        key="canvas",
    )

    if st.button("✅ Save Annotations"):
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            yolo_labels = []

            for obj in objects:
                if obj["type"] == "rect":
                    x1 = int(obj["left"])
                    y1 = int(obj["top"])
                    x2 = int(x1 + obj["width"])
                    y2 = int(y1 + obj["height"])
                    yolo_labels.append(to_yolo_format([x1, y1, x2, y2], img_w, img_h))

            label_filename = selected_file.rsplit(".", 1)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_filename)

            with open(label_path, "w") as f:
                f.write("\n".join(yolo_labels))

            st.success(f"✅ Saved {len(yolo_labels)} annotations to `{label_path}`")
        else:
            st.warning("⚠️ No annotations to save.")

# === Count Summary Section ===
st.write("---")
st.header("📊 Pizza Count Summary")

try:
    response = requests.get(API_URL)
    data = response.json()

    if data["status"] == "ok":
        count_data = data["data"]
        total = count_data.pop("total", 0)
        st.table(count_data)
        st.success(f"🎯 Total Pizzas Counted: {total}")
    else:
        st.warning(data.get("error", "No count data available."))
except Exception as e:
    st.error(f"❌ Could not fetch pizza count data: {e}")

st.write("---")
if st.button("🧠 Retrain model from feedback"):
    with st.spinner("Retraining model..."):
        try:
            subprocess.run([sys.executable, "../backend/retrain_from_feedback.py"], check=True)
            st.success("🎉 Retraining complete. Check 'retrained/pizza_model/' for weights.")
        except subprocess.CalledProcessError as e:
            st.error(f"Retraining failed: {e}")
