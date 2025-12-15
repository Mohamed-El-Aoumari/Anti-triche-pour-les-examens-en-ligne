import sys
import cv2
import os
import base64
import numpy as np
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
        
        # Dimensions par défaut
        self.width = 640
        self.height = 480
        
        self.extractors = initialize_extractors(self.width, self.height)

        self.rf_predictor = CheatingPredictor(
            model_path = os.path.join(root_path, 'models', 'rf_model.pkl'), 
            columns_path = os.path.join(root_path, 'models', 'model_columns.pkl')
        )
        self.time_manager = TimeWindowPredictor(fps=10, window_seconds=4, threshold=0.4)

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
            except Exception as e:
                db.session.rollback()
                print(f"Erreur DB: {e}")

    def process_frame(self, base64_image_string):
        try:
            if "," in base64_image_string:
                base64_image_string = base64_image_string.split(",")[1]
            
            img_data = base64.b64decode(base64_image_string)
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None: return {'status': 'error', 'msg': 'Empty image'}
            
            # Mise à jour dimensions
            h, w, _ = image.shape
            if w != self.width or h != self.height:
                self.width, self.height = w, h

            image = cv2.flip(image, 1)

            # Extraction & Prédiction
            all_features = features_extraction(image, self.extractors)
            instant_pred = self.rf_predictor.predict_one_frame(all_features)
            is_in_alert, current_rate, finished_alert = self.time_manager.process_frame(instant_pred)

            if finished_alert:
                self.save_alert_to_db(finished_alert)

            # Dessin
            if is_in_alert:
                cv2.rectangle(image, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(image, f"ALERTE ({current_rate:.0%})", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
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

            # Encodage retour
            _, buffer = cv2.imencode('.jpg', image)
            processed_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "status": "ok",
                "is_alert": bool(is_in_alert),
                "cheat_rate": float(current_rate),
                "image": "data:image/jpeg;base64," + processed_base64
            }

        except Exception as e:
            print(f"Erreur process: {e}")
            return {"status": "error"}