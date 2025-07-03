
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from main import app as plank_app
from main2 import app as squat_app

# Create parent app
parent_app = FastAPI(title="Unified Plank & Squat API")

# Enable CORS
parent_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount both apps under separate prefixes
parent_app.mount("/plank", plank_app)
parent_app.mount("/squat", squat_app)

@parent_app.get("/")
def read_root():
    return {
        "message": "Welcome to the Unified Posture Analysis API",
        "routes": {
            "plank": "/plank",
            "squat": "/squat"
        }
    }
