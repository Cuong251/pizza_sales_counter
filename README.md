# 🍕 Pizza Sales Counter System

This is an end-to-end intelligent video analytics system for counting pizza sales using YOLOv8 + DeepSORT. It supports:

- Real-time detection and tracking from video
- Streamlit UI for user feedback and annotation
- Model retraining based on feedback
- REST API to expose results

---

## 🚀 Quick Start with Docker Compose

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/pizza-sales-counter.git
cd pizza-sales-counter
```

### 2. Place Your Videos and Model
- Put your video files in `dataset/videos/`
- Put your trained model in `runs/detect/train4/weights/best.pt`

### 3. Start the System
```bash
docker-compose up --build
```

### 4. Access the App
- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI (counts): [http://localhost:8000/counts](http://localhost:8000/counts)

---

## 📂 Project Structure
```
pizza-sales-counter/
├── backend/
│   ├── api.py                  # FastAPI backend
│   └── retrain_from_feedback.py
├── frontend/
│   └── feedback_app.py        # Streamlit interface
├── dataset/
│   ├── videos/                # Input videos
│   ├── feedback_candidates/   # Frames to annotate
│   └── feedback_labels/       # YOLO-style user labels
├── runs/
│   └── detect/train4/weights/best.pt
├── tracker/                   # DeepSORT model
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
└── README.md
```

---

## 🎓 Requirements (Optional Outside Docker)
- Python 3.11+
- `pip install -r requirements.txt`

---

## 📚 Features
- YOLOv8 for pizza detection
- DeepSORT for object tracking
- Feedback interface with drawing boxes
- Trigger retraining from UI
- Result summary via API

---

## 📊 Result Format
```json
{
  "status": "ok",
  "data": {
    "video1.mp4": 42,
    "video2.mp4": 37,
    "total": 79
  }
}
```

---

## 🛠️ Future Improvements
- Upload images in UI
- Auto-label suggestions
- Export trained weights via UI

