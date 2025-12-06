import cv2
import time
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


def initialize_extractors(width, height):
    
    # Définition de la zone de script 
    sx = int(width * 0.20)
    sy = int(height * 0.42)
    sw = int(width * 0.60)
    sh = int(height * 0.42)
    script_area = (sx, sy, sw, sh)

    extractors = {
        "phone": PhoneDetector(),
        "face": FaceDetectionExtractor(),
        "hand": HandTrackingExtractor(),
        "head": HeadPositionExtractor(),
        "eye": EyeGazeExtractor(script_area_rect=script_area)
    }
    return extractors

def features_extraction(image, extractors):

    phone_features = extractors["phone"].process_image(image)
    face_features = extractors["face"].extract_face_features(image)
    hand_features = extractors["hand"].extract_hand_features(image)
    head_position_features = extractors["head"].extract_head_position(image)
    eye_gaze_features = extractors["eye"].process_image(image)

    features = {
        **phone_features,
        **face_features,
        **hand_features,
        **head_position_features,
        **eye_gaze_features
    }

    return features


'''
dans cette partie, on va generer des fichier csv a partir des images
dans le dossier "Images/N/*" N c'est le nom du dossier qui contient les images normales
pour chaque image on va extraire les features et les stocker dans un fichier csv nommé "0_features.csv"
dans le dossier "Images/T/*" T c'est le nom du dossier qui contient les images de triche
pour chaque image on va extraire les features et les stocker dans un fichier csv nommé "1_features.csv"
'''

if __name__ == "__main__":
    import os
    import pandas as pd

    image_dirs = {
        0: "Images/N/",
        1: "Images/T/"
    }

    for label, dir_path in image_dirs.items():
        all_features = []
        image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"{dir_path} est vide.")
            continue

        # Initialisation des extracteurs avec la taille de la première image
        sample_image = cv2.imread(os.path.join(dir_path, image_files[0]))
        height, width, _ = sample_image.shape
        extractors = initialize_extractors(width, height)

        for image_file in image_files:
            image_path = os.path.join(dir_path, image_file)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Erreur de chargement de l'image {image_path}.")
                continue

            features = features_extraction(image, extractors)
            features['image_file'] = image_file
            features['label'] = label
            all_features.append(features)

        # Save features to CSV dans le dossier Data_CSV
        df = pd.DataFrame(all_features)
        if not os.path.exists("Data_CSV"):
            os.makedirs("Data_CSV")
        csv_filename = f"{label}_features.csv"
        csv_path = os.path.join("Data_CSV", csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved features for {label} images to {csv_path}")