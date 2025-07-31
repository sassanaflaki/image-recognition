import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

def finger_status(hand_landmarks):
    """Returns a list of raised fingers [Thumb, Index, Middle, Ring, Pinky]"""
    fingers = []

    # Thumb (horizontal check)
    if hand_landmarks.landmark[FINGER_TIPS[0]].x < hand_landmarks.landmark[FINGER_TIPS[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other four fingers (vertical check)
    for i in range(1, 5):
        if hand_landmarks.landmark[FINGER_TIPS[i]].y < hand_landmarks.landmark[FINGER_TIPS[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        finger_text = "No hand detected"
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = finger_status(hand_landmarks)
                raised_fingers = [FINGER_NAMES[i] for i, val in enumerate(fingers) if val == 1]
                finger_text = f"Fingers: {sum(fingers)} ({', '.join(raised_fingers)})"

        cv2.putText(img, finger_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return img

st.title("Real-Time Hand & Finger Detection")
st.write("Shows which fingers are raised (Thumb, Index, Middle, Ring, Pinky).")
webrtc_streamer(key="finger-counter", video_transformer_factory=VideoTransformer)
