import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

print("Booting up the ISL Real-Time Translator...")

actions = np.array([
    'Alright', 'Good Morning', 'Good afternoon',
    'Good evening', 'Good night', 'Hello',
    'How are you', 'Pleased', 'Thank you'
])

print("Loading model...")
model = tf.keras.models.load_model('isl_custom_model.h5') 
print("Model loaded successfully!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_math = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Left': 
                lh = hand_math
            else: 
                rh = hand_math
    return np.concatenate([lh, rh])

print("Starting webcam")
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

sequence = []
current_sign = "Waiting for sign"
confidence = 0.0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-40:]

        if len(sequence) == 40:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_idx = np.argmax(res)
            confidence = res[predicted_idx]
            
            if confidence > 0.85:
                current_sign = actions[predicted_idx]

        cv2.rectangle(frame, (0,0), (640, 60), (0, 0, 0), -1)
        
        cv2.putText(frame, f"{current_sign}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('ISL Real-Time Translator', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("Closing application...")
            break

finally:

    print("Shutting down and turning off camera...")
    cap.release()
    cv2.destroyAllWindows()