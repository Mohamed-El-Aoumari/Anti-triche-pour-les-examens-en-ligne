import cv2
import time
from FeaturesExtraction.Phone_detection import PhoneDetector
from FeaturesExtraction.Face_detection import FaceDetectionExtractor
from FeaturesExtraction.Hand_tracking import HandTrackingExtractor
from FeaturesExtraction.Head_position import HeadPositionExtractor
from FeaturesExtraction.Eye_gaze_tracking import EyeGazeExtractor

#########################################################
## Ici, on va developper le code qui fait l'extraction ##
## des différentes features en utilisant les modules   ##
##    définis dans le dossier FeaturesExtraction       ##
#########################################################


def initialize_extractors(width, height):
    
    # Définition de la zone de script 
    sx = int(width * 0.20)
    sy = int(height * 0.42)
    sw = int(width * 0.60)
    sh = int(height * 0.42)
    script_area = (sx, sy, sw, sh)

    extractors = {
        "phone": PhoneDetector(),
        "face": FaceDetectionExtractor(),
        "hand": HandTrackingExtractor(),
        "head": HeadPositionExtractor(),
        "eye": EyeGazeExtractor(script_area_rect=script_area)
    }
    return extractors

def features_extraction(image, extractors):

    phone_features = extractors["phone"].process_image(image)
    face_features = extractors["face"].extract_face_features(image)
    hand_features = extractors["hand"].extract_hand_features(image)
    head_position_features = extractors["head"].extract_head_position(image)
    eye_gaze_features = extractors["eye"].process_image(image)

    features = {
        **phone_features,
        **face_features,
        **hand_features,
        **head_position_features,
        **eye_gaze_features
    }

    return features

#Test
image = cv2.imread("Screenshot_3.png")
features = features_extraction(image, initialize_extractors(image.shape[1], image.shape[0]))
print(features)

'''

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    ret, first_frame = cap.read()
    h, w = first_frame.shape[:2]
    extractors_dict = initialize_extractors(w, h)

    # Calcul de FPS
    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # On passe le dictionnaire d'extracteurs déjà chargés
        features = features_extraction(frame, extractors_dict)
        print(features)

        # Calcul FPS et affichage
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

'''