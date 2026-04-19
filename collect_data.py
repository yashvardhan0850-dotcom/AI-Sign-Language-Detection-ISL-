import cv2
import numpy as np
import os
import mediapipe as mp
import time

print("ISL Custom Dataset Generator!")
teammate_name = input("Enter your first name (No spaces): ").strip()

DATA_PATH = 'Custom_ISL_Data' 
actions = np.array(['Alright', 'Good Morning', 'Good afternoon', 'Good evening', 'Good night', 'Hello', 'How are you', 'Pleased', 'Thank you'])

no_sequences = 10 
sequence_length = 40 

for action in actions: 
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    rh = np.zeros(21 * 3)
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_math = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if handedness.classification[0].label == 'Left': lh = hand_math
            else: rh = hand_math
    return np.concatenate([lh, rh])
#(1) for extrenal webcam ex. iriun, (0) for laptop webcam
cap = cv2.VideoCapture(1)

print("\nTurning on camera")

for action_idx, action in enumerate(actions):
    
    rest_time = 30 if action_idx > 0 else 5
    
    for i in range(rest_time, 0, -1):
        ret, frame = cap.read()
        if not ret: break
        
        cv2.putText(frame, f"NEXT SIGN: '{action}'", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Rest your arms! Starting in: {i}s", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Data Collection', frame)
        cv2.waitKey(1000) 

    for sequence in range(no_sequences):
        
        for i in range(4, 0, -1):
            ret, frame = cap.read()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
            cv2.putText(frame, f"Get ready for '{action}' ({sequence+1}/{no_sequences})", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, str(i), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 6, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000) 

        window = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            cv2.rectangle(frame, (0,0), (640, 50), (0, 255, 0), -1)
            cv2.putText(frame, f"RECORDING '{action}' | Frame {frame_num+1}/40", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            
            keypoints = extract_keypoints(results)
            window.append(keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
        npy_path = os.path.join(DATA_PATH, action, f"{action}_{teammate_name}_{sequence}.npy")
        np.save(npy_path, np.array(window))

cap.release()
cv2.destroyAllWindows()
print("\n DATA COLLECTION COMPLETE!")
