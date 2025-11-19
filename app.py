import streamlit as st
import cv2
import time
from PIL import Image
from utils import load_model, beep_alert, predict_posture_on_frame, analyze_image
import mediapipe as mp

# ================= CONFIG =================
MODEL_PATH = "posture_model_best.pkl"
BAD_THRESHOLD = 5.0

# ================= LOAD MODEL =================
@st.cache_resource
def get_model(path):
    return load_model(path)

clf = get_model(MODEL_PATH)
mp_pose = mp.solutions.pose

# ================= STREAMLIT UI =================
st.set_page_config(page_title="AI Posture App", page_icon="🧍", layout="centered")
st.title("🧍 AI Posture Detection & Classification")

tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "📷 Take Photo", "🖥️ Live Webcam"])

# ---------- TAB 1: UPLOAD IMAGE ----------
with tab1:
    uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Original Image", use_column_width=True)
        if st.button("Analyze Uploaded Image"):
            with st.spinner("Analyzing posture..."):
                out_img, label, conf = analyze_image(pil_img, clf)
            if label is None:
                st.warning("No pose detected.")
            else:
                st.image(out_img, caption="Pose & Prediction", use_column_width=True)
                st.success(f"Prediction: **{label.upper()}** ({conf*100:.1f}%)")
                if label == "bad":
                    beep_alert()
                    st.error("Detected BAD posture – please correct your posture.")

# ---------- TAB 2: CAMERA ----------
with tab2:
    camera_img = st.camera_input("Take a picture")
    if camera_img:
        pil_cam_img = Image.open(camera_img)
        st.image(pil_cam_img, caption="Captured Image", use_column_width=True)
        if st.button("Analyze Camera Image"):
            with st.spinner("Analyzing posture..."):
                out_img, label, conf = analyze_image(pil_cam_img, clf)
            if label is None:
                st.warning("No pose detected.")
            else:
                st.image(out_img, caption="Pose & Prediction", use_column_width=True)
                st.success(f"Prediction: **{label.upper()}** ({conf*100:.1f}%)")
                if label == "bad":
                    beep_alert()
                    st.error("Detected BAD posture – please correct your posture.")

# ---------- TAB 3: LIVE WEBCAM ----------
with tab3:
    st.write("Real-time posture detection using your webcam.")
    if st.button("Start Webcam"):
        FRAME_WINDOW = st.empty()
        info_box = st.empty()
        timer_box = st.empty()
        cap = cv2.VideoCapture(0)
        bad_start_time = None
        bad_beeped = False

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    info_box.error("Cannot access webcam.")
                    break
                frame = cv2.flip(frame, 1)
                frame, label, conf = predict_posture_on_frame(frame, clf, pose)
                now = time.time()

                # ----- bad posture timing logic -----
                bad_duration = 0.0
                if label == "bad":
                    if bad_start_time is None:
                        bad_start_time = now
                    bad_duration = now - bad_start_time
                    if bad_duration >= BAD_THRESHOLD and not bad_beeped:
                        beep_alert()
                        bad_beeped = True
                    cv2.putText(frame, f"BAD for {bad_duration:.1f}s", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    bad_start_time = None
                    bad_beeped = False

                # ----- main label -----
                if label is None:
                    info_box.warning("No pose detected...")
                    timer_box.empty()
                else:
                    color = (0, 255, 0) if label == "good" else (0, 0, 255)
                    cv2.putText(frame, f"{label.upper()} ({conf*100:.1f}%)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    info_box.success(f"Posture: {label.upper()}  ({conf*100:.1f}%)")
                    if label == "bad":
                        timer_box.warning(f"BAD posture for {bad_duration:.1f}s "
                                          f"(beep at {BAD_THRESHOLD:.0f}s)")
                    else:
                        timer_box.info("Posture is GOOD ✅")

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not cap.isOpened():
                    break
        cap.release()
