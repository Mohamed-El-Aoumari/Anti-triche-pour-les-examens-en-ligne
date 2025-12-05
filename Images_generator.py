import cv2
import os
import math


def frames_extractor(source, dest, target_fps=1):

    # Les catégories basées sur notre structure de dossiers
    categories = ['N', 'T']
    
    for category in categories:

        path_source = os.path.join(source, category)
        path_dest = os.path.join(dest, category) 
        
        os.makedirs(path_dest, exist_ok=True) 
        

        print(f"--- Traitement du dossier : {category} ---")
        
        # Lister les fichiers vidéos
        files = [f for f in os.listdir(path_source) if f.lower().endswith(('.mp4'))]
        
        for video_file in files:
            video_path = os.path.join(path_source, video_file)
            print(f"Traitement de la vidéo : {video_file}")
            
            # Ouvrir la vidéo
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Erreur: Impossible d'ouvrir {video_file}")
                continue
            
            # Récupérer les FPS (Frames Per Second) originaux de la vidéo
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Si la vidéo est à 30 fps et on veut 2 fps, on capture une image toutes les 15 frames (30/2)
            if original_fps > 0:
                frame_interval = math.ceil(original_fps / target_fps)
            else:
                frame_interval = 15 # Valeur par défaut si on ne peut pas obtenir les FPS
            
            frame_count = 0
            saved_count = 0
            
            while True:
                success, frame = cap.read()
                
                if not success:
                    break 
                
                # On sauvegarde seulement si le numéro de frame correspond à l'intervalle
                if frame_count % frame_interval == 0:
                    # On crée un nom de fichier unique pour chaque image
                    base_name = os.path.splitext(video_file)[0]
                    img_name = f"{base_name}_img_{saved_count}.jpg"
                    output_path = os.path.join(path_dest, img_name)
                    
                    cv2.imwrite(output_path, frame)
                    saved_count += 1
                
                frame_count += 1
            
            cap.release()
            print(f" -> {saved_count} images extraites pour {video_file}")

    print("\nFin du traitement de toutes les vidéos.")



# Lancement du script

if __name__ == "__main__":

    dossier_source = "Data"
    dossier_destination = "Images"
    
    frames_extractor(dossier_source, dossier_destination, target_fps=1)