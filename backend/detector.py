import os
import cv2
from ultralytics import YOLO
from deep_sort_tracker import DeepSort

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "..", "dataset", "videos")
FEEDBACK_DIR = os.path.join(BASE_DIR, "..", "dataset", "feedback_candidates")
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Path to YOLO detection model
YOLO_MODEL_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "tracker", "weights", "yolov5s.pt"
))

# Path to DeepSort encoder model
DEEPSORT_MODEL_PATH = os.path.abspath(os.path.join(
    BASE_DIR, "..", "tracker", "deep_sort", "deep", "checkpoint", "resnet18.pth"
))

# === Initialize Models ===
print("[INFO] Using YOLO model:", YOLO_MODEL_PATH)
print("[INFO] Using DeepSort reID model:", DEEPSORT_MODEL_PATH)

model = YOLO(r"E:\pizza_sales_counter\runs\detect\train4\weights\best.pt")  # Replace if needed
deepsort = DeepSort(model_path=DEEPSORT_MODEL_PATH)

# === Frame skipping config ===
FRAME_SKIP = 5  # Process 1 of every 5 frames

def process_video(video_path):
    print(f"[INFO] Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    unique_pizzas = set()
    frame_id = 0
    saved_frame_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id % FRAME_SKIP != 0:
            continue

        # Run detection
        results = model.predict(source=frame, conf=0.5, iou=0.5, stream=False)[0]

        detections = []
        found_pizza = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name.lower() != "pizza":
                continue

            found_pizza = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            detections.append([x, y, w, h, conf])

        # Save frame for feedback (limit to 30 per video)
        if found_pizza and saved_frame_count < 30:
            save_path = os.path.join(FEEDBACK_DIR, f"{video_name}_frame{frame_id}.jpg")
            cv2.imwrite(save_path, frame)
            saved_frame_count += 1

        # Run tracking
        tracked_objects = deepsort.update_tracks(detections, frame)
        for obj in tracked_objects:
            unique_pizzas.add(obj.track_id)

    cap.release()
    return len(unique_pizzas)


def process_all_videos(video_folder=VIDEO_DIR):
    print(f"[INFO] Scanning: {video_folder}")
    results = {}
    total = 0

    if not os.path.exists(video_folder):
        print("[ERROR] Video folder not found:", video_folder)
        return {}

    for file in os.listdir(video_folder):
        if file.endswith((".mp4", ".avi", ".mov")):
            full_path = os.path.join(video_folder, file)
            count = process_video(full_path)
            results[file] = count
            total += count

    results["total"] = total
    return results


if __name__ == "__main__":
    output = process_all_videos()

    # Optional: Save to JSON for API
    import json
    output_path = os.path.join(BASE_DIR, "pizza_counts.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== 🍕 Pizza Count Summary ===")
    for video, count in output.items():
        print(f"{video}: {count} pizzas")
