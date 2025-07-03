# main.py
from fastapi import FastAPI
from detector import PlankDetector
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
detector = PlankDetector()

# Allow CORS for mobile app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Plank Form API is Live"}

@app.post("/start")
def start_detection():
    detector.start()
    return {"status": "started"}

@app.post("/stop")
def stop_detection():
    detector.stop()
    return {"status": "stopped"}

@app.get("/status")
def get_status():
    return detector.get_status()
