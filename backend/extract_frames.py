import cv2
import os

# Absolute path of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Correct paths based on your structure
VIDEO_DIR = os.path.join(BASE_DIR, "..", "dataset", "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "dataset", "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)
frame_interval = 30  # One frame per second if 30 FPS

for video_name in os.listdir(VIDEO_DIR):
    if video_name.endswith(".mp4"):
        video_path = os.path.join(VIDEO_DIR, video_name)
        cap = cv2.VideoCapture(video_path)
        i = 0
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if i % frame_interval == 0:
                img_path = os.path.join(OUTPUT_DIR, f"{video_name}_{i}.jpg")
                cv2.imwrite(img_path, frame)
                saved += 1
            i += 1

        cap.release()
        print(f"[{video_name}] Saved {saved} frames.")
