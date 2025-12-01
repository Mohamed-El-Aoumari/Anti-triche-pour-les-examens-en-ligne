import cv2
import mediapipe as mp
import numpy as np
from math import dist

# ----- Face Mesh -----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Lip landmark indices (MediaPipe FaceMesh)
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

def mouth_ratio(landmarks, img_w, img_h):
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]

    upper = np.array([upper.x * img_w, upper.y * img_h])
    lower = np.array([lower.x * img_w, lower.y * img_h])
    left = np.array([left.x * img_w, left.y * img_h])
    right = np.array([right.x * img_w, right.y * img_h])

    vertical = dist(upper, lower)
    horizontal = dist(left, right)
    return vertical / horizontal, (upper + lower) / 2  # Return center of mouth

# ----- Hands -----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

THRESHOLD = 0.28
SPEAKING_FRAMES = 0
FRAME_LIMIT = 5
CHEATING_DISTANCE = 50  # pixels, adjust if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)

    speaking = False
    cheating = False

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        ratio, mouth_center = mouth_ratio(face.landmark, w, h)

        # Detect speaking
        if ratio > THRESHOLD:
            SPEAKING_FRAMES += 1
        else:
            SPEAKING_FRAMES = max(0, SPEAKING_FRAMES - 1)

        if SPEAKING_FRAMES > FRAME_LIMIT:
            speaking = True

        cv2.putText(frame, f"Mouth Ratio: {ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Detect hands in frame
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Use tip of middle finger (landmark 12)
            finger_tip = hand_landmarks.landmark[12]
            finger_point = np.array([finger_tip.x * w, finger_tip.y * h])

            # Check distance to mouth
            if face_results.multi_face_landmarks:
                if dist(finger_point, mouth_center) < CHEATING_DISTANCE:
                    cheating = True
                    break  # No need to check other hands

    # Status label
    if cheating:
        cv2.putText(frame, "CHEATING!", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    elif speaking:
        cv2.putText(frame, "Speaking", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    else:
        cv2.putText(frame, "Silent", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.imshow("Exam Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
