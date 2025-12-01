########################################################
##   Dans cette etape, on va creer une classe pour    ##
## extraire les caracteristiques de la main a partir  ##
## des images en utilisant MediaPipe Hands et OpenCV. ##
########################################################

'''
Les caracteristiques de la main extraites sont :
    ** hand_count : Integer, Number of hands detected in the frame.
    ** left_hand_x, left_hand_y : Float, X, Y coordinates of the left hand. Zero if not detected.
    ** right_hand_x, right_hand_y : Float, X, Y coordinates of the right hand. Zero if not detected.
    ** hand_obj_interaction : Integer (Binary), Indicator: 1 if hand is interacting with an object, 0 otherwise.
'''

#######################################################
## La sortie sera un dictionnaire de cette format :  ##
## {                                                 ##
##    'hand_count': 2,                               ##
##    'left_hand_x': 120.0,                          ##
##    'left_hand_y': 250.0,                          ##
##    'right_hand_x': 300.0,                         ##
##    'right_hand_y': 260.0,                         ##
##    'hand_obj_interaction': 1                      ##
## }                                                 ##
#######################################################

import cv2
import mediapipe as mp
import numpy as np

class HandTrackingExtractor:
    def __init__(self, static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5):
        """
        Initialise le detecteur de mains MediaPipe
        
        Args:
            static_image_mode: Si True, traite les images comme des images statiques
            max_num_hands: Nombre maximum de mains à détecter
            min_detection_confidence: Confiance minimale pour la détection
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        
    def extract_hand_features(self, image):
        """
        Extrait les caracteristiques des mains d'une image
        
        Args:
            image: Image BGR (format OpenCV)
            
        Returns:
            dict: Dictionnaire contenant les caracteristiques des mains
        """
        # Convertir l'image BGR en RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # Initialiser les valeurs par défaut
        features = {
            'hand_count': 0,
            'left_hand_x': 0.0,
            'left_hand_y': 0.0,
            'right_hand_x': 0.0,
            'right_hand_y': 0.0,
            'hand_obj_interaction': 0
        }
        
        if results.multi_hand_landmarks and results.multi_handedness:
            features['hand_count'] = len(results.multi_hand_landmarks)
            
            # Traiter chaque main détectée
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Determiner si c'est la main gauche ou droite
                hand_label = handedness.classification[0].label
                
                # Calculer la position moyenne de la main (paume)
                landmarks_array = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
                hand_center = np.mean(landmarks_array, axis=0)
                
                # Convertir les coordonnées normalisées en coordonnées pixels
                height, width = image.shape[:2]
                hand_x = hand_center[0] * width
                hand_y = hand_center[1] * height
                
                if hand_label == "Left":
                    features['left_hand_x'] = float(hand_x)
                    features['left_hand_y'] = float(hand_y)
                elif hand_label == "Right":
                    features['right_hand_x'] = float(hand_x)
                    features['right_hand_y'] = float(hand_y)
            
            # Detection basique d'interaction avec un objet
            features['hand_obj_interaction'] = self._detect_object_interaction(results.multi_hand_landmarks, image)
        
        return features
    
    def _detect_object_interaction(self, hand_landmarks_list, image):
        """
        Detecte l'interaction entre les mains et des objets
        Version simplifiée - à adapter selon vos besoins
        
        Args:
            hand_landmarks_list: Liste des landmarks des mains
            image: Image originale
            
        Returns:
            int: 1 si interaction détectée, 0 sinon
        """
        if not hand_landmarks_list:
            return 0
        
        for hand_landmarks in hand_landmarks_list:
            landmarks = hand_landmarks.landmark
            
            # Calculer la distance entre le pouce et l'index
            thumb_tip = np.array([landmarks[self.mp_hands.HandLandmark.THUMB_TIP].x,
                                landmarks[self.mp_hands.HandLandmark.THUMB_TIP].y])
            index_tip = np.array([landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            
            distance = np.linalg.norm(thumb_tip - index_tip)
            
            # Si la distance est petite, la main pourrait tenir un objet
            if distance < 0.05:
                return 1
        
        return 0


# Le code de test
if __name__ == "__main__":
    # Initialisation
    hand_extractor = HandTrackingExtractor()
    
    # Capturer video
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extraire les caractéristiques
        features = hand_extractor.extract_hand_features(frame)
        
        # Afficher les caractéristiques sur l'image
        cv2.putText(frame, f"Hands: {features['hand_count']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Left: ({features['left_hand_x']:.1f}, {features['left_hand_y']:.1f})", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right: ({features['right_hand_x']:.1f}, {features['right_hand_y']:.1f})", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Interaction: {features['hand_obj_interaction']}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Hand Tracking', frame)

        print(features)  # Afficher les caractéristiques dans la console
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()
