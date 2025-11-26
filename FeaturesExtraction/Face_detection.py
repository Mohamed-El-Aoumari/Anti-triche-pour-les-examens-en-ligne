######################################################
##  Dans cette etape, on va creer une classe pour   ##
## extraire les caracteristiques de visage a partir ##
## des images en utilisant MediaPipe Face Detection ## 
##                et OpenCV.                        ##
######################################################

''' 
Les caracteristiques de visage extraites sont :
    ** face_present : Integer (Binary) Indicator: 1 if a face is detected, 0 otherwise. 0, 1
    ** no_of_face : Integer, Number of faces detected in the frame.
    ** face_x, face_y : Float, Coordinates of the top-left corner of the detected face bounding box.
    ** face_w, face_h : Float, Width and height of the detected face bounding box.
    ** left_eye_x, left_eye_y : Float, coordinates of the left eye’s center.
    ** right_eye_x, right_eye_y : Float, coordinates of the right eye’s center.
    ** nose_tip_x, nose_tip_y : Float, coordinates of the nose tip.
    ** mouth_x, mouth_y : Float, coordinates of the mouth center.
    ** face_conf : Float, Confidence score (0-100) of the face detection.
'''

######################################################
## La sortie sera un dictionnaire de cette format : ##
## {                                                ##
##    'face_present': 1,                            ##
##    'no_of_face': 1,                              ##
##    'face_x': 100.0,                              ##
##    'face_y': 150.0,                              ##
##    'face_w': 200.0,                              ##
##    'face_h': 200.0,                              ##
##    'left_eye_x': 140.0,                          ##
##    'left_eye_y': 200.0,                          ##
##    'right_eye_x': 220.0,                         ##
##    'right_eye_y': 200.0,                         ##
##    'nose_tip_x': 180.0,                          ##
##    'nose_tip_y': 250.0,                          ##
##    'mouth_x': 180.0,                             ##
##    'mouth_y': 300.0,                             ##
##    'face_conf': 95.0                             ##
## }                                                ##
######################################################

class FaceDetectionExtractor:
    def __init__(self):
        pass

    def extract_face_features(self, image):
        pass