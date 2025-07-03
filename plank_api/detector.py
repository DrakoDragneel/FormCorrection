# detector.py
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import time
from collections import deque
import threading

class PlankDetector:
    def __init__(self):
        self.cap = None
        self.running = False
        self.last_status = None
        self.correct_time = 0
        self.incorrect_time = 0
        self.correct_start = None
        self.incorrect_start = None
        self.prediction_history = deque(maxlen=5)
        self.prediction_threshold = 0.6
        self.frame = None

        with open("model/LR_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open("model/input_scaler.pkl", "rb") as f2:
            self.scaler = pickle.load(f2)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_class(self, pred):
        return {0: "C", 1: "H", 2: "L"}.get(pred)

    def extract_keypoints(self, results):
        keypoints = []
        lm = results.pose_landmarks.landmark
        important = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
            "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
        ]
        for name in important:
            point = lm[self.mp_pose.PoseLandmark[name].value]
            keypoints.extend([point.x, point.y, point.z, point.visibility])
        return keypoints

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_detection)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.pose.close()

        # Reset timers & state
        self.last_status = None
        self.correct_time = 0
        self.incorrect_time = 0
        self.correct_start = None
        self.incorrect_start = None
        self.prediction_history.clear()

    def get_status(self):
        # Calculate live timers
        correct_total = self.correct_time
        incorrect_total = self.incorrect_time

        if self.last_status == "C" and self.correct_start:
            correct_total += time.time() - self.correct_start
        elif self.last_status in ["H", "L"] and self.incorrect_start:
            incorrect_total += time.time() - self.incorrect_start

        return {
            "form": self.last_status or "Analyzing...",
            "correct_time": int(correct_total),
            "incorrect_time": int(incorrect_total)
        }

    def _run_detection(self):
        self.cap = cv2.VideoCapture(0)

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.pose_landmarks:
                try:
                    row = self.extract_keypoints(results)
                    X = pd.DataFrame([row])
                    X_scaled = self.scaler.transform(X)

                    pred = self.model.predict(X_scaled)[0]
                    prob = self.model.predict_proba(X_scaled)[0]

                    label = self.get_class(pred)
                    if prob[np.argmax(prob)] >= self.prediction_threshold:
                        self.prediction_history.append(label)

                    if len(self.prediction_history) == 5:
                        majority = max(set(self.prediction_history), key=self.prediction_history.count)
                        if majority != self.last_status:
                            if majority == "C":
                                self.correct_start = time.time()
                                if self.incorrect_start:
                                    self.incorrect_time += time.time() - self.incorrect_start
                                self.incorrect_start = None
                            else:
                                self.incorrect_start = time.time()
                                if self.correct_start:
                                    self.correct_time += time.time() - self.correct_start
                                self.correct_start = None
                            self.last_status = majority

                except Exception as e:
                    print("Prediction error:", e)

            time.sleep(0.01)
