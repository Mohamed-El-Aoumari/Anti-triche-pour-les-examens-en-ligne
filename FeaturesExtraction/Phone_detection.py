######################################################
##  Dans cette etape, on va creer une classe pour   ##
##  detecter la presence de telephone en utilisant  ## 
##                  YOLOv7 et OpenCV.               ##
######################################################


'''
Les caracteristiques du telephone extraites sont :
    ** phone_present : Integer (Binary), Indicator: 1 if a phone is detected, 0 otherwise.
    ** phone_loc_x, phone_loc_y : Float, X, Y coordinates of the detected phone. Zero if not detected.
    ** phone_conf : Float, Confidence score (0-1) of the phone detection.
'''

######################################################
## La sortie sera un dictionnaire de cette format : ##
## {                                                ##
##    'phone_present': 1,                           ##
##    'phone_loc_x': 150.0,                         ##
##    'phone_loc_y': 200.0,                         ##
##    'phone_conf': 0.85                            ##
## }                                                ##
######################################################

import cv2
from ultralytics import YOLO
import numpy as np

class PhoneDetector:
    def __init__(self, confidence_threshold=0.4):
        # Chargement du modèle YOLOv8
        self.model = YOLO('yolov8n.pt') 
        
        self.conf_thresh = confidence_threshold

        self.PHONE_CLASS_ID = 67 # ID de la classe "phone" dans COCO dataset

    def process_image(self, image):

        results = self.model(image, verbose=False, conf=self.conf_thresh)
        
        # Dictionnaire par défaut
        output = {
            'phone_present': 0,
            'phone_loc_x': 0.0,
            'phone_loc_y': 0.0,
            'phone_conf': 0.0
        }

        # results[0] contient les boîtes détectées pour la première image
        detections = results[0].boxes

        if detections:
            indices = (detections.cls == self.PHONE_CLASS_ID).nonzero(as_tuple=True)[0]

            if len(indices) > 0:
                best_idx = indices[0] # Premier index trouvé
                
                box = detections.xyxy[best_idx].cpu().numpy() # [x1, y1, x2, y2]
                conf = float(detections.conf[best_idx].cpu())
                
                # Calcul du centre
                center_x = (box[0] + box[2]) / 2
                center_y = (box[1] + box[3]) / 2
                
                output['phone_present'] = 1
                output['phone_loc_x'] = float(center_x)
                output['phone_loc_y'] = float(center_y)
                output['phone_conf'] = conf

        return output


# Le code de test
if __name__ == "__main__":
    detector = PhoneDetector()
    
    cap = cv2.VideoCapture(0) 
    
    print("Test Détection Téléphone - Appuyez sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        data = detector.process_image(frame)
        print(data)

        # Dessin pour visualisation
        if data['phone_present'] == 1:
            x, y = int(data['phone_loc_x']), int(data['phone_loc_y'])
            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(frame, f"PHONE {data['phone_conf']:.2f}", (x, y-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Phone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
