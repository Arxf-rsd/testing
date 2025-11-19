import joblib
import threading
import winsound
import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ================= UTILITIES =================
def load_model(path):
    """Load a trained posture model (joblib)"""
    return joblib.load(path)

def beep_alert():
    """Short beep in a separate thread so it doesn't freeze UI."""
    def _beep():
        winsound.Beep(1000, 300)  # 1000 Hz, 0.3 sec
    threading.Thread(target=_beep, daemon=True).start()

def landmarks_to_feature_vector(landmarks):
    """Extract 7 key points (x, y, z) for posture classification."""
    key_ids = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    features = []
    for lm_id in key_ids:
        lm = landmarks[lm_id.value]
        features.extend([lm.x, lm.y, lm.z])
    return features

def predict_posture_on_frame(frame, clf, pose):
    """Run Mediapipe + model prediction on a frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return frame, None, 0.0

    mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    feats = np.array(landmarks_to_feature_vector(result.pose_landmarks.landmark)).reshape(1, -1)
    probs = clf.predict_proba(feats)[0]
    classes = clf.classes_
    idx = probs.argmax()
    label = classes[idx]
    conf = probs[idx]
    return frame, label, conf

def analyze_image(pil_img, clf):
    """Analyze uploaded or camera image."""
    img = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return img, None, None
        mp_drawing.draw_landmarks(img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        feats = np.array(landmarks_to_feature_vector(results.pose_landmarks.landmark)).reshape(1, -1)
        probs = clf.predict_proba(feats)[0]
        classes = clf.classes_
        best_idx = probs.argmax()
        pred_label = classes[best_idx]
        pred_conf = probs[best_idx]

        color = (0, 255, 0) if pred_label == "good" else (0, 0, 255)
        text = f"{pred_label.upper()} ({pred_conf*100:.1f}%)"
        cv2.putText(img_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out_img, pred_label, pred_conf
