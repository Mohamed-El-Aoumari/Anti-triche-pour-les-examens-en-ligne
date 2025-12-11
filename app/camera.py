import sys
import cv2
import os
from models import db, CheatAlertSegment, ExamSession
from modules.predictor import CheatingPredictor
from modules.time_predictor import TimeWindowPredictor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import features_extraction, initialize_extractors 

root_path = os.path.dirname(os.path.abspath(__file__))


class VideoCamera(object):
    def __init__(self, app, session_id):
        self.app = app
        self.session_id = session_id
        self.video = cv2.VideoCapture(0)

        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Si la caméra n'est pas encore prête, on met des valeurs par défaut
        if self.width == 0 or self.height == 0:
            self.width, self.height = 640, 480

        self.extractors = initialize_extractors(self.width, self.height)

        self.rf_predictor = CheatingPredictor(
            model_path = root_path + r'\models\rf_model.pkl', 
            columns_path = root_path + r'\models\model_columns.pkl'
        )
        self.time_manager = TimeWindowPredictor(fps=10, window_seconds=4, threshold=0.4)

    def __del__(self):
        self.video.release()

    def save_alert_to_db(self, alert_data):
        with self.app.app_context():            
            new_alert = CheatAlertSegment(
                session_id=self.session_id,
                start_time=alert_data['start_time'],
                end_time=alert_data['end_time'],
                avg_cheat_rate=alert_data['avg_score']
            )
            
            db.session.add(new_alert)
            
            session = ExamSession.query.get(self.session_id)
            if session:
                session.cheat_probability = min(100.0, session.cheat_probability + (alert_data['avg_score']*10))
            
            try:
                db.session.commit()
                print(f"Alerte sauvegardée pour la session {self.session_id}")
            except Exception as e:
                db.session.rollback()
                print(f"Erreur DB: {e}")

    def get_frame(self):
        success, image = self.video.read()
        if not success: return None
        
        image = cv2.flip(image, 1)

        # L'extraction des features
        all_features = features_extraction(image, self.extractors)

        # Prédiction instantanée
        instant_pred = self.rf_predictor.predict_one_frame(all_features)

        # 
        is_in_alert, current_rate, finished_alert = self.time_manager.process_frame(instant_pred)

        if finished_alert:
            self.save_alert_to_db(finished_alert)

        '''
        
        # Dessin Zone Script (On accède à l'extracteur 'eye' stocké dans le dictionnaire)
        eye_extractor = self.extractors["eye"]
        sx, sy, sw, sh = eye_extractor.script_area
        
        # On vérifie si 'gaze_on_script' est vrai dans le gros dictionnaire
        is_on_script = all_features.get('gaze_on_script', 0) == 1
        color_box = (0, 255, 0) if is_on_script else (0, 0, 255)
        
        cv2.rectangle(image, (sx, sy), (sx+sw, sy+sh), color_box, 1)
        cv2.putText(image, "ZONE EXAMEN", (sx, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)
        '''
        # Alerte visuelle
        if is_in_alert:
            cv2.rectangle(image, (0, 0), (int(self.width), 60), (0, 0, 255), -1)
            cv2.putText(image, f"ALERTE TRICHE ({current_rate:.0%})", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Dessin Téléphone si détecté
            if all_features.get('phone_present', 0) == 1:
                px = int(all_features.get('phone_loc_x', 0))
                py = int(all_features.get('phone_loc_y', 0))
                if px > 0:
                    cv2.putText(image, "PHONE", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        elif current_rate > 0.3:
            cv2.putText(image, f"Suspicion: {current_rate:.0%}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(image, "Surveillance Active", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()