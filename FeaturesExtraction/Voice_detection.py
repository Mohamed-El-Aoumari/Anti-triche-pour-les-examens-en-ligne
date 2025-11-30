import cv2
import mediapipe as mp
import numpy as np
from math import dist

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
    # Get positions
    upper = landmarks[UPPER_LIP]
    lower = landmarks[LOWER_LIP]
    left = landmarks[LEFT_MOUTH]
    right = landmarks[RIGHT_MOUTH]

    # Convert normalized coords → pixels
    upper = np.array([upper.x * img_w, upper.y * img_h])
    lower = np.array([lower.x * img_w, lower.y * img_h])
    left = np.array([left.x * img_w, left.y * img_h])
    right = np.array([right.x * img_w, right.y * img_h])

    # Distance vertical / horizontal
    vertical = dist(upper, lower)
    horizontal = dist(left, right)

    return vertical / horizontal  # Mouth openness ratio


cap = cv2.VideoCapture(0)

THRESHOLD = 0.28   # adjust if needed
SPEAKING_FRAMES = 0
FRAME_LIMIT = 5    # number of frames to confirm speaking

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    speaking = False

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        ratio = mouth_ratio(face.landmark, w, h)

        # Detect speaking
        if ratio > THRESHOLD:
            SPEAKING_FRAMES += 1
        else:
            SPEAKING_FRAMES = max(0, SPEAKING_FRAMES - 1)

        if SPEAKING_FRAMES > FRAME_LIMIT:
            speaking = True

        # Display ratio
        cv2.putText(frame, f"Mouth Ratio: {ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Status label
    if speaking:
        cv2.putText(frame, "Speaking", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    else:
        cv2.putText(frame, "Silent", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)

    cv2.imshow("Mouth Movement Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

