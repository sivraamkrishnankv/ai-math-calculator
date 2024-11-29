import cvzone, cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st

# Streamlit configuration
st.set_page_config(layout="wide")
output_text = "" 
col1, col2 = st.columns([2,1])  
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with col2:
    output_text_area = st.title("Answer")
    output_text_area = st.subheader("")


st.sidebar.markdown("### Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.5)
tracking_confidence = st.sidebar.slider("Tracking Confidence", min_value=0.1, max_value=1.0, value=0.5)

# Configure Google Generative AI
genai.configure(api_key="<apikey>")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector
detector = HandDetector(staticMode=True, maxHands=1, modelComplexity=1, detectionCon=detection_confidence, minTrackCon=tracking_confidence)

# Functions
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    return None

def draw(info, prev_pos, canvas):
    fingers, lmlist = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmlist[8][0:2]
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 0, 0, 0, 0]:  # Clear canvas
        canvas = np.zeros_like(img)
        prev_pos = None
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve the problem ", pil_image])
        return response.text

# Main loop
prev_pos = None
canvas = None

while run:
    success, img = cap.read()
    # img = cv2.flip(img, 1)

    if not success:
        st.error("Failed to capture image.")
        break

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmlist = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        output_text = sendToAI(model, canvas, fingers)
        if output_text:
            output_text_area.text(output_text)

    image_combined = cv2.addWeighted(img, 0.65, canvas, 0.35, 0)
    FRAME_WINDOW.image(image_combined, channels="BGR")

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
