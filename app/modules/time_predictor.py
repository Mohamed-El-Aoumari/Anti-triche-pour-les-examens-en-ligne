import time
from collections import deque
from datetime import datetime

class TimeWindowPredictor:
    def __init__(self, fps=10, window_seconds=15, threshold=0.4):
        # Taille du buffer = Nombre d'images à garder en mémoire
        self.buffer_size = int(fps * window_seconds)
        self.buffer = deque(maxlen=self.buffer_size)
        self.threshold = threshold
        
        # Gestion d'état pour la base de données (Machine à états)
        self.in_alert_mode = False       # Est-ce qu'on est actuellement en train de tricher ?
        self.alert_start_time = None     # Quand est-ce que ça a commencé ?
        self.current_alert_scores = []   # Liste des scores pour calculer la moyenne de l'incident
        self.min_alert_duration = 5.0    # Durée min (en sec) pour valider une alerte

    def process_frame(self, prediction_0_or_1):

        # On ajoute la prédiction dans le buffer glissant
        self.buffer.append(prediction_0_or_1)
        
        # On calcule le taux actuel sur la fenêtre
        # Si le buffer n'est pas plein (début de test), on calcule sur ce qu'on a
        if len(self.buffer) == 0:
            current_rate = 0.0
        else:
            current_rate = sum(self.buffer) / len(self.buffer)

        finished_alert_data = None

        # La logique de décision
        if current_rate >= self.threshold:
            # CAS 1 : LE TAUX DÉPASSE LE SEUIL (Zone Rouge)
            
            if not self.in_alert_mode:
                # Démarrage d'une nouvelle alerte
                self.in_alert_mode = True
                self.alert_start_time = datetime.utcnow()
                self.current_alert_scores = []
            
            # On continue d'enregistrer les scores tant qu'on est au-dessus du seuil
            self.current_alert_scores.append(current_rate)
            
        else:
            # CAS 2 : LE TAUX EST SOUS LE SEUIL (Zone Verte) 
            
            if self.in_alert_mode:
                # Fin potentielle de l'alerte
                duration = (datetime.utcnow() - self.alert_start_time).total_seconds()
                
                # On ne sauvegarde que si l'alerte a duré assez longtemps
                # Cela évite de créer une alerte si l'étudiant tourne la tête 1 seconde
                if duration >= self.min_alert_duration:
                    avg_final = sum(self.current_alert_scores) / len(self.current_alert_scores)
                    
                    finished_alert_data = {
                        'start_time': self.alert_start_time,
                        'end_time': datetime.utcnow(),
                        'avg_score': float(avg_final),
                        'duration': duration
                    }

                # Réinitialisation de l'état
                self.in_alert_mode = False
                self.alert_start_time = None
                self.current_alert_scores = []

        return self.in_alert_mode, current_rate, finished_alert_data