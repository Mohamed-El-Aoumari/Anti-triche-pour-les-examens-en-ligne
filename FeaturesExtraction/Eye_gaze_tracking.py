##################################################
##    Dans cette etape, on va creer une classe  ##
## pour extraire les caracteristiques du regard ##
##   des yeux a partir des images en utilisant  ##
##    MediaPipe Face Mesh, OpenCV et NumPy.     ##
##################################################

'''
Les caracteristiques du regard des yeux extraites sont :
    ** gaze_on_script : Integer (Binary), Indicator: 1 if gaze is directed towards the defined 'script' area, 0 otherwise.
    ** gaze_direction : Categorical, General direction of gaze. 'center', 'bottom_right', 'bottom_left', 'None'
    ** gazePoint_x, gazePoint_y : Float, X, Y coordinates of the estimated gaze point on the screen/frame.
    ** pupil_left_x, pupil_left_y : Float, X, Y coordinates of the left pupil.
    ** pupil_right_x, pupil_right_y : Float, X, Y coordinates of the right pupil.
'''

######################################################
## La sortie sera un dictionnaire de cette format : ##
## {                                                ##
##    'gaze_on_script': 1,                          ##
##    'gaze_direction': 'center',                   ##
##    'gazePoint_x': 400.0,                         ##
##    'gazePoint_y': 300.0,                         ##
##    'pupil_left_x': 180.0,                        ##
##    'pupil_left_y': 220.0,                        ##
##    'pupil_right_x': 220.0,                       ##
##    'pupil_right_y': 220.0                        ##
## }                                                ##
######################################################



import cv2
import mediapipe as mp
import numpy as np


class EyeGazeExtractor:
    def __init__(self, script_area_rect=None):

        # Initialisation de MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) 
        
        self.script_area = script_area_rect # (x, y, width, height)

        # Indices des landmarks clés
        self.LEFT_IRIS = [474, 475, 476, 477]  # Landmarks de l'iris gauche
        self.RIGHT_IRIS = [469, 470, 471, 472] # Landmarks de l'iris droit
        self.LEFT_PUPIL_CENTER = 468           # Centre de la pupille gauche
        self.RIGHT_PUPIL_CENTER = 473          # Centre de la pupille droite
        
        # Indices pour solvePnP (Estimation de pose)
        self.FACE_3D_POINTS = np.array([
            (0.0, 0.0, 0.0),             # Nez
            (0.0, -330.0, -65.0),        # Menton
            (-225.0, 170.0, -135.0),     # Coin Oeil G
            (225.0, 170.0, -135.0),      # Coin Oeil D
            (-150.0, -150.0, -125.0),    # Bouche G
            (150.0, -150.0, -125.0)      # Bouche D
        ], dtype=np.float64)

        # Indices correspondants dans MediaPipe (Nez:1, Menton:199, Coin Oeil G:33, Coin Oeil D:263, Bouche G:61, Bouche D:291)
        self.FACE_2D_INDICES = [1, 199, 33, 263, 61, 291]

    def process_image(self, image):
        """
        dans cette méthode, on traite une image et on retourne le dictionnaire de caractéristiques.
        """
        h, w, c = image.shape
        
        # Conversion couleur pour MediaPipe
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        # Dictionnaire par défaut
        output_data = {
            'gaze_on_script': 0,
            'gaze_direction': 'None',
            'gazePoint_x': 0.0,
            'gazePoint_y': 0.0,
            'pupil_left_x': 0.0,
            'pupil_left_y': 0.0,
            'pupil_right_x': 0.0,
            'pupil_right_y': 0.0
        }

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Extraction des Pupilles (Coordonnées Pixel) 
                # Left Pupil
                lp = landmarks[self.LEFT_PUPIL_CENTER]
                px_l, py_l = lp.x * w, lp.y * h
                output_data['pupil_left_x'] = float(px_l)
                output_data['pupil_left_y'] = float(py_l)

                # Right Pupil
                rp = landmarks[self.RIGHT_PUPIL_CENTER]
                px_r, py_r = rp.x * w, rp.y * h
                output_data['pupil_right_x'] = float(px_r)
                output_data['pupil_right_y'] = float(py_r)

                # Estimation du point de regard (Gaze Point) via solvePnP 
                focal_length = w
                cam_matrix = np.array([
                    [focal_length, 0, w / 2],
                    [0, focal_length, h / 2],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1)) 

                # Récupération des points 2D de l'image
                image_points = []
                for idx in self.FACE_2D_INDICES:
                    lm = landmarks[idx]
                    image_points.append([lm.x * w, lm.y * h])
                image_points = np.array(image_points, dtype=np.float64)

                # Résolution PnP pour obtenir rotation et translation
                success, rvec, tvec = cv2.solvePnP(
                    self.FACE_3D_POINTS, 
                    image_points, 
                    cam_matrix, 
                    dist_coeffs
                )

                if success:
                    nose_end_point3D = np.array([[0.0, 0.0, 500.0]]) # Point devant le nez
                    nose_end_point2D, _ = cv2.projectPoints(
                        nose_end_point3D, rvec, tvec, cam_matrix, dist_coeffs
                    ) # Projection 3D -> 2D

                    p1 = (int(image_points[0][0]), int(image_points[0][1])) # Point du nez
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) # Point projeté
                    
                    # Le point projeté est notre estimation du 'gazePoint' sur l'écran
                    gaze_x, gaze_y = float(p2[0]), float(p2[1])
                    output_data['gazePoint_x'] = gaze_x
                    output_data['gazePoint_y'] = gaze_y

                    # Détermination de la Direction et Vérification de la zone de script
                    script_x, script_y, script_w, script_h = self.script_area
                    
                    # Vérification zone script
                    is_on_script = (script_x < gaze_x < script_x + script_w) and (script_y < gaze_y < script_y + script_h)
                    
                    output_data['gaze_on_script'] = 1 if is_on_script else 0

                    # Détermination de la direction du regard

                    cx, cy = w // 2, h // 2 # Centre de l'écran
                    dx = gaze_x - cx        # Déplacement horizontal par rapport au centre
                    dy = gaze_y - cy        # Déplacement vertical par rapport au centre
                    
                    # Seuils 
                    threshold_x = w * 0.15 # 15% de la largeur
                    threshold_y = h * 0.15 # 15% de la hauteur

                    direction = 'center'
                    if dy > threshold_y: # Regarde vers le bas
                        if dx > threshold_x:
                            direction = 'bottom_right'
                        elif dx < -threshold_x:
                            direction = 'bottom_left'
                        else:
                            direction = 'center'
                    elif dy < -threshold_y: # Regarde vers le haut
                        if dx > threshold_x:
                            direction = 'top_right'
                        elif dx < -threshold_x:
                            direction = 'top_left'
                        else:
                            direction = 'center'

                    output_data['gaze_direction'] = direction

        return output_data


# Exemple d'utilisation :

'''

image = cv2.imread('22.jpg') 

h, w = image.shape[:2]
sx = int(w * 0.20)
sy = int(h * 0.42)
sw = int(w * 0.60)
sh = int(h * 0.42)
script_area = (sx, sy, sw, sh)

extractor = EyeGazeExtractor(script_area_rect=script_area)
features = extractor.process_image(image)
print(features)

'''

# Le code de test

if __name__ == "__main__":
    
    # Capturer video
    cap = cv2.VideoCapture(0)

    # On definit la zone de script
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sx = int(w * 0.20)
    sy = int(h * 0.42)
    sw = int(w * 0.60)
    sh = int(h * 0.42)
    script_area = (sx, sy, sw, sh)

    # Initialisation
    eye_gaze_extractor = EyeGazeExtractor(script_area_rect=script_area)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        features = eye_gaze_extractor.process_image(frame)
        
        # Affichage des caractéristiques sur l'image
        cv2.putText(frame, f"Gaze on Script: {features['gaze_on_script']}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze Direction: {features['gaze_direction']}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze Point: ({int(features['gazePoint_x'])}, {int(features['gazePoint_y'])})", 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.circle(frame, (int(features['pupil_left_x']), int(features['pupil_left_y'])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(features['pupil_right_x']), int(features['pupil_right_y'])), 5, (0, 0, 255), -1)
        
        cv2.imshow('Eye Gaze Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()