import pandas as pd
import joblib
import os

class CheatingPredictor:
    def __init__(self, model_path, columns_path):
        self.model = None
        self.model_columns = []

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
            except Exception as e:
                print(f"Erreur chargement modèle : {e}")
        else:
            print(f"Fichier modèle introuvable : {model_path}")

        # Chargement de la liste des colonnes utilisées lors de l'entraînement
        if os.path.exists(columns_path):
            try:
                self.model_columns = joblib.load(columns_path)
            except Exception as e:
                print(f"Erreur chargement colonnes : {e}")
        else:
            print(f"Fichier colonnes introuvable : {columns_path}")

    def predict_one_frame(self, features):
        if self.model is None or not self.model_columns:
            return 0

        try:
            # On nettoie le dictionnaire : Si une valeur est True/False, elle devient 1/0
            clean_features = {}
            for k, v in features.items():
                # On supprime le 'label' s'il existe
                if k == 'label':
                    continue
                
                # On convertit les booléens en int (True->1, False->0)
                if isinstance(v, bool):
                    clean_features[k] = int(v)
                else:
                    clean_features[k] = v

            # Conversion du dict nettoyé en DataFrame
            df_row = pd.DataFrame([clean_features])

            df_row.fillna('unknown', inplace=True)

            # Encodage One-Hot (get_dummies)
            df_encoded = pd.get_dummies(df_row)

            expected_cols = [c for c in self.model_columns if c != 'label']

            # ALIGNEMENT DES COLONNES
            df_encoded = df_encoded.reindex(columns=expected_cols, fill_value=0)


            # Prédiction
            prediction = self.model.predict(df_encoded)
            
            return int(prediction[0])

        except Exception as e:
            print(f"Erreur prédiction : {e}")
            return 0