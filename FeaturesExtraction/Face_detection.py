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


import cv2
import mediapipe as mp
import numpy as np

class FaceDetectionExtractor:
    def __init__(self):
        """
        Initialise le détecteur de visage MediaPipe avec Face Mesh
        pour obtenir des landmarks détaillés (yeux, nez, bouche)
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        # Initialisation du Face Mesh pour les landmarks détaillés
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )
        
        # Initialisation du Face Detection pour la confiance et bounding box
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Indices des landmarks MediaPipe pour les caractéristiques faciales
        self.LEFT_EYE_INDEX = 33   # Centre de l'œil gauche
        self.RIGHT_EYE_INDEX = 263  # Centre de l'œil droit
        self.NOSE_TIP_INDEX = 1     # Pointe du nez
        self.MOUTH_CENTER_INDEX = 13  # Centre de la bouche

    def extract_face_features(self, image):
        """
        Extrait les caractéristiques faciales d'une image
        
        Args:
            image: Image BGR (format OpenCV) ou RGB
            
        Returns:
            dict: Dictionnaire contenant toutes les caractéristiques faciales
        """
        # Initialisation du dictionnaire de sortie avec valeurs par défaut
        features = {
            'face_present': 0,
            'no_of_face': 0,
            'face_x': 0.0,
            'face_y': 0.0,
            'face_w': 0.0,
            'face_h': 0.0,
            'left_eye_x': 0.0,
            'left_eye_y': 0.0,
            'right_eye_x': 0.0,
            'right_eye_y': 0.0,
            'nose_tip_x': 0.0,
            'nose_tip_y': 0.0,
            'mouth_x': 0.0,
            'mouth_y': 0.0,
            'face_conf': 0.0
        }
        
        if image is None or image.size == 0:
            return features
        
        # Conversion BGR vers RGB si nécessaire
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        h, w = image_rgb.shape[:2]
        
        # Détection avec Face Detection pour la bounding box et confiance
        detection_results = self.face_detection.process(image_rgb)
        
        # Détection avec Face Mesh pour les landmarks
        mesh_results = self.face_mesh.process(image_rgb)
        
        # Si aucun visage détecté
        if not detection_results.detections and not mesh_results.multi_face_landmarks:
            return features
        
        # Nombre de visages détectés
        no_of_faces = 0
        if detection_results.detections:
            no_of_faces = len(detection_results.detections)
        elif mesh_results.multi_face_landmarks:
            no_of_faces = len(mesh_results.multi_face_landmarks)
        
        features['face_present'] = 1
        features['no_of_face'] = no_of_faces
        
        # Extraction des informations du premier visage détecté
        if detection_results.detections:
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Conversion des coordonnées relatives en pixels
            features['face_x'] = float(bbox.xmin * w)
            features['face_y'] = float(bbox.ymin * h)
            features['face_w'] = float(bbox.width * w)
            features['face_h'] = float(bbox.height * h)
            
            # Score de confiance (0-1 converti en 0-100)
            features['face_conf'] = float(detection.score[0] * 100)
        
        # Extraction des landmarks (yeux, nez, bouche)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            
            # Œil gauche
            left_eye = landmarks[self.LEFT_EYE_INDEX]
            features['left_eye_x'] = float(left_eye.x * w)
            features['left_eye_y'] = float(left_eye.y * h)
            
            # Œil droit
            right_eye = landmarks[self.RIGHT_EYE_INDEX]
            features['right_eye_x'] = float(right_eye.x * w)
            features['right_eye_y'] = float(right_eye.y * h)
            
            # Pointe du nez
            nose_tip = landmarks[self.NOSE_TIP_INDEX]
            features['nose_tip_x'] = float(nose_tip.x * w)
            features['nose_tip_y'] = float(nose_tip.y * h)
            
            # Centre de la bouche
            mouth = landmarks[self.MOUTH_CENTER_INDEX]
            features['mouth_x'] = float(mouth.x * w)
            features['mouth_y'] = float(mouth.y * h)
            
            # Si pas de bounding box de Face Detection, on l'estime depuis les landmarks
            if features['face_w'] == 0.0 or features['face_h'] == 0.0:
                all_x = [lm.x * w for lm in landmarks]
                all_y = [lm.y * h for lm in landmarks]
                features['face_x'] = float(min(all_x))
                features['face_y'] = float(min(all_y))
                features['face_w'] = float(max(all_x) - min(all_x))
                features['face_h'] = float(max(all_y) - min(all_y))
                features['face_conf'] = 85.0  # Confiance par défaut
        
        return features
    
    def __del__(self):
        """Libération des ressources"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
        if hasattr(self, 'face_detection'):
            self.face_detection.close()


