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
    """
    Extrait la position et l'orientation de la tête à partir d'une image.
    Utilise MediaPipe Face Mesh pour détecter les points faciaux et calculer
    les angles de rotation (pitch, yaw, roll).
    """
    
    def __init__(self):
        # Initialiser MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Définir les seuils pour la classification de la pose
        self.pitch_threshold = 15  # degrés
        self.yaw_threshold = 15    # degrés
    
    def extract_head_position(self, image):
        """
        Extrait les caractéristiques de position de la tête depuis une image.
        
        Args:
            image: Image numpy array (BGR format de OpenCV)
            
        Returns:
            dict: Dictionnaire contenant head_pose, head_pitch, head_roll, head_yaw
        """
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
        
        # Extraire les points 3D clés pour calculer l'orientation
        # Points de référence sur le visage (indices MediaPipe Face Mesh)
        landmarks_points = []
        indices = [1, 33, 263, 61, 291, 199]  # Nez, œil gauche, œil droit, etc.
        
        for idx in indices:
            landmark = face_landmarks.landmark[idx]
            landmarks_points.append([landmark.x * w, landmark.y * h])
        
        landmarks_points = np.array(landmarks_points, dtype=np.float32)
        
        # Points 3D du modèle de visage générique
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nez
            (-30.0, -125.0, -30.0),      # Œil gauche
            (30.0, -125.0, -30.0),       # Œil droit
            (-20.0, -70.0, -125.0),      # Coin gauche de la bouche
            (20.0, -70.0, -125.0),       # Coin droit de la bouche
            (0.0, 0.0, -90.0)            # Menton
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
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            landmarks_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
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
        
        # Déterminer la pose catégorielle
        head_pose = self._classify_head_pose(pitch, yaw)
        
        return {
            'head_pose': head_pose,
            'head_pitch': float(pitch),
            'head_roll': float(roll),
            'head_yaw': float(yaw)
        }
    
    def _rotation_matrix_to_euler_angles(self, R):
        """
        Convertit une matrice de rotation en angles d'Euler (pitch, yaw, roll).
        
        Args:
            R: Matrice de rotation 3x3
            
        Returns:
            tuple: (pitch, yaw, roll) en degrés
        """
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
        """
        Classifie la pose de la tête en catégories : 'forward', 'down', 'left', 'right'.
        
        Args:
            pitch: Angle de rotation autour de l'axe X (haut/bas)
            yaw: Angle de rotation autour de l'axe Y (gauche/droite)
            
        Returns:
            str: Catégorie de pose
        """
        # Priorité à yaw pour gauche/droite
        if abs(yaw) > self.yaw_threshold:
            if yaw > 0:
                return 'right'
            else:
                return 'left'
        
        # Ensuite pitch pour haut/bas
        if pitch > self.pitch_threshold:
            return 'down'
        
        # Par défaut, si angles faibles
        return 'forward'
    


