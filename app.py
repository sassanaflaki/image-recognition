import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector
detector = HandDetector(maxHands=1, detectionCon=0.7)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        hands, img = detector.findHands(img)  # with draw

        finger_text = "No hand detected"
        if hands:
            fingers = detector.fingersUp(hands[0])
            raised_fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            up_names = [raised_fingers[i] for i, val in enumerate(fingers) if val == 1]
            finger_text = f"Fingers: {sum(fingers)} ({', '.join(up_names)})"

        cv2.putText(img, finger_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

st.title("Real-Time Hand & Finger Detection (Cloud Friendly)")
st.write("Shows which fingers are raised using cvzone.")
webrtc_streamer(key="finger-counter", video_transformer_factory=VideoTransformer)
