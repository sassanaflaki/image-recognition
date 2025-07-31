import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load Haar Cascade for hand detection
# (This is a simple approach, not as advanced as Mediapipe)
hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_hand.xml")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hand_cascade = hand_cascade

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect hands
        hands = self.hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(hands) == 0:
            cv2.putText(img, "No hand detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            for (x, y, w, h) in hands:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, f"Hands detected: {len(hands)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

st.title("Real-Time Hand Detection (No Mediapipe)")
st.write("Detects hands using OpenCV Haar cascades (no finger counting).")
webrtc_streamer(key="hand-detector", video_transformer_factory=VideoTransformer)
