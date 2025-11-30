########################################################
##  Dans cette etape, on va creer une classe pour     ##
##    extraire les caracteristiques de la tete a      ##
## partir des images en utilisant MediaPipe Face Mesh,## 
##               OpenCV et NumPy.                     ##
########################################################

'''
Les caracteristiques de la tete extraites sont :
    ** head_pose : Categorical: 'forward', 'down', 'left', 'right', 'None'
    ** head_pitch : Float, Head rotation around the X-axis in degrees.
    ** head_roll : Float, Head rotation around the Z-axis (tilt left/right).
    ** head_yaw : Float, Head rotation around the Y-axis (left/right turn).
'''

########################################################
## La sortie sera un dictionnaire de cette format :   ##
## {                                                  ##
##    'head_pose': 'forward',                         ##
##    'head_pitch': 10.5,                             ##
##    'head_roll': -5.2,                              ##
##    'head_yaw': 15.0                                ##
## }                                                  ##
########################################################

import cv2
import mediapipe as mp
import numpy as np

class HeadPositionExtractor:

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Angles plus stricts pour une classification plus précise
        self.pitch_down_threshold = 10  # Seuil pour détecter "down"
        self.yaw_left_threshold = 12    # Seuil pour détecter "left"
        self.yaw_right_threshold = 12   # Seuil pour détecter "right"
        self.forward_tolerance = 8      # Tolérance pour "forward"
    
    def extract_head_position(self, image):
      
        # Convertir BGR vers RGB pour MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Détecter les landmarks faciaux
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            # Aucun visage détecté
            return {
                'head_pose': 'None',
                'head_pitch': 0.0,
                'head_roll': 0.0,
                'head_yaw': 0.0
            }
        
        face_landmarks = results.multi_face_landmarks[0]
   
        landmarks_points = []
        indices = [
            1,      # Pointe du nez
            33,     # Coin externe de l'œil gauche
            263,    # Coin externe de l'œil droit
            61,     # Coin interne de l'œil gauche
            291,    # Coin interne de l'œil droit
            199,    # Partie inférieure du nez
            10,     # Front
            152     # Menton
        ]
        
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            landmarks_points.append([landmark.x * w, landmark.y * h])
        
        landmarks_points = np.array(landmarks_points, dtype=np.float32)
        
        # Points 3D du modèle de visage générique (modèle optimisé)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Pointe du nez
            (-50.0, -70.0, -40.0),       # Coin externe œil gauche
            (50.0, -70.0, -40.0),        # Coin externe œil droit
            (-30.0, -70.0, -30.0),       # Coin interne œil gauche
            (30.0, -70.0, -30.0),        # Coin interne œil droit
            (0.0, -30.0, -60.0),         # Partie inférieure du nez
            (0.0, -100.0, -70.0),        # Front
            (0.0, 30.0, -80.0)           # Menton
        ], dtype=np.float32)
        
        # Paramètres de la caméra (approximation)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Coefficients de distorsion (assumés nuls)
        dist_coeffs = np.zeros((4, 1))
        
        # Résoudre PnP pour obtenir les vecteurs de rotation et translation
        # Utilisation de SOLVEPNP_EPNP pour plus de stabilité
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            landmarks_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            return {
                'head_pose': 'None',
                'head_pitch': 0.0,
                'head_roll': 0.0,
                'head_yaw': 0.0
            }
        
        # Convertir le vecteur de rotation en matrice de rotation
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculer les angles d'Euler (pitch, yaw, roll)
        pitch, yaw, roll = self._rotation_matrix_to_euler_angles(rotation_matrix)
        
        # Appliquer des filtres pour stabiliser les angles
        pitch = self._stabilize_angle(pitch)
        yaw = self._stabilize_angle(yaw)
        roll = self._stabilize_angle(roll)
        
        # Déterminer la pose catégorielle
        head_pose = self._classify_head_pose(pitch, yaw)
        
        return {
            'head_pose': head_pose,
            'head_pitch': float(pitch),
            'head_roll': float(roll),
            'head_yaw': float(yaw)
        }
    
    def _rotation_matrix_to_euler_angles(self, R):
        # Calculer les angles d'Euler à partir de la matrice de rotation
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        
        singular = sy < 1e-6
        
        if not singular:
            pitch = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[1, 0], R[0, 0])
        else:
            pitch = np.arctan2(-R[1, 2], R[1, 1])
            yaw = np.arctan2(-R[2, 0], sy)
            roll = 0
        
        # Convertir en degrés
        pitch = np.degrees(pitch)
        yaw = np.degrees(yaw)
        roll = np.degrees(roll)
        
        return pitch, yaw, roll
    
    def _classify_head_pose(self, pitch, yaw):
        # Stratégie de classification hiérarchique
        
        # 1. Vérifier si la tête regarde vers le bas (pitch positif = vers le bas)
        if pitch > self.pitch_down_threshold:
            return 'down'
        
        # 2. Vérifier les rotations gauche/droite (yaw)
        # yaw négatif = vers la gauche, yaw positif = vers la droite
        if yaw < -self.yaw_left_threshold:
            return 'left'
        elif yaw > self.yaw_right_threshold:
            return 'right'
        
        # 3. Si les angles sont dans la tolérance, c'est "forward"
        if abs(pitch) <= self.forward_tolerance and abs(yaw) <= self.forward_tolerance:
            return 'forward'
        
        # 4. Cas ambigus : choisir selon l'angle dominant
        if abs(pitch) > abs(yaw):
            if pitch > 0:
                return 'down'
            else:
                return 'forward'  # Légèrement vers le haut = forward
        else:
            if yaw < 0:
                return 'left'
            else:
                return 'right'
    
    def _stabilize_angle(self, angle):
        # Arrondir les angles très petits à 0
        if abs(angle) < 2.0:
            return 0.0
        return angle
    
'''
# Exemple d'utilisation :
if __name__ == "__main__":
    
    # Capturer video
    cap = cv2.VideoCapture(0)
    head_position_extractor = HeadPositionExtractor()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        features = head_position_extractor.extract_head_position(frame)

        cv2.putText(frame, f"Head Pose: {features['head_pose']}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {features['head_pitch']:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {features['head_roll']:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {features['head_yaw']:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(features)
        
        cv2.imshow('Head Position Extraction', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
'''
