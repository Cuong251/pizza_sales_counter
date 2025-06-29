from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os

app = FastAPI()

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Path to count results ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COUNT_FILE = os.path.join(BASE_DIR, "pizza_counts.json")

# === Root endpoint ===
@app.get("/")
def read_root():
    return {"message": "🍕 Pizza Sales Counter API is running."}

# === Count results endpoint ===
@app.get("/counts")
def get_pizza_counts():
    if not os.path.exists(COUNT_FILE):
        return {"status": "error", "message": "No count results found. Please run detector.py first."}

    try:
        with open(COUNT_FILE, "r") as f:
            data = json.load(f)
        return {"status": "ok", "data": data}
    except Exception as e:
        return {"status": "error", "message": f"Failed to read count file: {str(e)}"}

