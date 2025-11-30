import cv2
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

import cv2
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

def features_extraction(image_path):
    image = cv2.imread(image_path)
    # On definit la zone de script
    h, w = image.shape[:2]
    sx = int(w * 0.20)
    sy = int(h * 0.42)
    sw = int(w * 0.60)
    sh = int(h * 0.42)
    script_area = (sx, sy, sw, sh)


    # Initialiser les extracteurs de features
    phone_detector = PhoneDetector()
    face_detector = FaceDetectionExtractor()
    hand_tracker = HandTrackingExtractor()
    head_position_extractor = HeadPositionExtractor()
    eye_gaze_extractor = EyeGazeExtractor(script_area_rect=script_area)

    # Extraire les features
    phone_features = phone_detector.process_image(image)
    face_features = face_detector.extract_face_features(image)
    hand_features = hand_tracker.extract_hand_features(image)
    head_position_features = head_position_extractor.extract_head_position(image)
    eye_gaze_features = eye_gaze_extractor.process_image(image)

    features = {
        **phone_features,
        **face_features,
        **hand_features,
        **head_position_features,
        **eye_gaze_features
    }

    return features


# Teste

print(features_extraction("Screenshot_3.png"))



