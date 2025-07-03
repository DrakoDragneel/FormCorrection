from fastapi import FastAPI, UploadFile, File
import shutil, uuid, os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import math
import collections

app = FastAPI()

# Load trained model
with open("model/LR_model.pkl", "rb") as f:
    count_model = pickle.load(f)

# Constants
IMPORTANT_LMS = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE"
]

headers = ["label"] + [f"{lm.lower()}_{axis}" for lm in IMPORTANT_LMS for axis in ["x", "y", "z", "v"]]
mp_pose = mp.solutions.pose

def extract_important_keypoints(results):
    landmarks = results.pose_landmarks.landmark
    return np.array([
        [landmarks[mp_pose.PoseLandmark[lm].value].x,
         landmarks[mp_pose.PoseLandmark[lm].value].y,
         landmarks[mp_pose.PoseLandmark[lm].value].z,
         landmarks[mp_pose.PoseLandmark[lm].value].visibility]
        for lm in IMPORTANT_LMS
    ]).flatten().tolist()

def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def analyze_foot_knee_placement(results, stage, foot_thresh, knee_thresh, vis_thresh):
    out = {"foot_placement": -1, "knee_placement": -1}
    lm = results.pose_landmarks.landmark
    required = [
        mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
        mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
    ]
    if any(lm[i.value].visibility < vis_thresh for i in required):
        return out

    shoulder_width = calculate_distance(
        [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    )
    foot_width = calculate_distance(
        [lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x, lm[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y],
        [lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x, lm[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
    )
    knee_width = calculate_distance(
        [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    )

    # Foot placement
    foot_ratio = round(foot_width / shoulder_width, 1)
    min_f, max_f = foot_thresh
    out["foot_placement"] = 0 if min_f <= foot_ratio <= max_f else 1 if foot_ratio < min_f else 2

    # Knee placement
    knee_ratio = round(knee_width / foot_width, 1)
    kmin, kmax = knee_thresh.get(stage, (0, 10))
    if kmin <= knee_ratio <= kmax:
        out["knee_placement"] = 0
    elif knee_ratio < kmin:
        out["knee_placement"] = 1
    else:
        out["knee_placement"] = 2

    return out

@app.get("/")
def root():
    return {"message": "Squat Form API is running"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    filename = f"temp_{uuid.uuid4().hex}.mp4"
    with open(filename, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(filename)
    stage = ""
    reps = 0
    foot_buffer = collections.deque(maxlen=5)
    knee_buffer = collections.deque(maxlen=5)

    FOOT_THRESH = [1.2, 2.8]
    KNEE_THRESH = {
        "up": [0.5, 1.0], "middle": [0.7, 1.0], "down": [0.7, 1.1]
    }

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.pose_landmarks:
                try:
                    row = extract_important_keypoints(results)
                    X = pd.DataFrame([row], columns=headers[1:])
                    pred = count_model.predict(X)[0]
                    pred = "down" if pred == 0 else "up"
                    prob = count_model.predict_proba(X)[0].max()

                    if pred == "down" and prob >= 0.7:
                        stage = "down"
                    elif stage == "down" and pred == "up" and prob >= 0.7:
                        stage = "up"
                        reps += 1

                    analysis = analyze_foot_knee_placement(results, stage, FOOT_THRESH, KNEE_THRESH, 0.6)
                    foot_buffer.append(analysis["foot_placement"])
                    knee_buffer.append(analysis["knee_placement"])
                except Exception as e:
                    print(f"⚠️ Frame error: {e}")
                    continue

    cap.release()
    os.remove(filename)

    foot_mode = max(set(foot_buffer), key=foot_buffer.count, default=-1)
    knee_mode = max(set(knee_buffer), key=knee_buffer.count, default=-1)

    foot_status = ["Correct", "Too tight", "Too wide", "UNK"][foot_mode] if foot_mode != -1 else "UNK"
    knee_status = ["Correct", "Too tight", "Too wide", "UNK"][knee_mode] if knee_mode != -1 else "UNK"

    return {
        "message": "Processed successfully",
        "total_reps": reps,
        "foot_status": foot_status,
        "knee_status": knee_status
    }
