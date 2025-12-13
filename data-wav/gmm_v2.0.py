import numpy as np
import librosa
import librosa.effects
from sklearn.mixture import GaussianMixture
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION SIMPLIFIÉE ---
SR = 16000           
N_MFCC = 13          # Retour à 13 (Standard robuste)
HOP_LEN = 512        
GMM_COMPONENTS = 16  # Réduit à 16 pour éviter d'apprendre le bruit

def preprocess_audio(path):
    try:
        y, _ = librosa.load(path, sr=SR)
        y = librosa.util.normalize(y)
        # VAD Stricte : On coupe agressivement le silence
        intervals = librosa.effects.split(y, top_db=20) 
        if len(intervals) == 0: return None
        y_speech = np.concatenate([y[start:end] for start, end in intervals])
        if len(y_speech) < 0.5 * SR: return None
        return y_speech
    except:
        return None

def extract_features(y):
    # On garde MFCC + Delta (Vitesse)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC, hop_length=HOP_LEN)
    mfcc_delta = librosa.feature.delta(mfcc)
    features = np.vstack([mfcc, mfcc_delta])
    # Normalisation CMVN essentielle
    features = (features - np.mean(features, axis=1, keepdims=True)) / (np.std(features, axis=1, keepdims=True) + 1e-6)
    return features.T

def train_system(train_files, val_file):
    print("\n🔹 PHASE 1: ENTRAÎNEMENT & CALIBRAGE...")
    
    # 1. Training
    feats_list = []
    for path in train_files:
        y = preprocess_audio(path)
        if y is not None: feats_list.append(extract_features(y))
        
    if not feats_list: raise ValueError("Aucune donnée d'entraînement valide.")
    X_train = np.vstack(feats_list)
    
    gmm = GaussianMixture(n_components=GMM_COMPONENTS, covariance_type='diag', random_state=42, max_iter=200)
    gmm.fit(X_train)
    
    # 2. Calibrage sur me3.wav (Validation)
    y_val = preprocess_audio(val_file)
    if y_val is None: raise ValueError("Fichier de validation invalide.")
    
    val_feats = extract_features(y_val)
    val_scores = gmm.score_samples(val_feats)
    
    # --- LOGIQUE CRITIQUE ---
    # On définit le seuil de score (Log-Likelihood)
    # On prend le percentile 10 : les 10% de TA voix les plus moches sont rejetés.
    score_threshold = np.percentile(val_scores, 10)
    
    # On calcule quel est TON taux de rejet "normal" avec ce seuil
    # (Ce sera mathématiquement proche de 10%)
    my_reject_count = np.sum(val_scores < score_threshold)
    my_rejection_rate = my_reject_count / len(val_scores)
    
    # LA MARGE DE SÉCURITÉ :
    # Si tu as 10% de rejet, on autorise jusqu'à 15% max.
    # Tout ce qui est au-dessus est considéré comme étranger.
    max_allowed_rejection = my_rejection_rate + 0.05 # Marge de 5%
    
    print(f"   ✅ Modèle entraîné.")
    print(f"   🎯 Seuil de Score (Log-Likelihood) : {score_threshold:.2f}")
    print(f"   📊 Ton Taux de Rejet de base : {my_rejection_rate*100:.1f}%")
    print(f"   🔒 LIMITE MAXIMALE AUTORISÉE : {max_allowed_rejection*100:.1f}%")
    
    return gmm, score_threshold, max_allowed_rejection

def test_files(gmm, score_threshold, max_reject_rate, files):
    print("\n🔹 PHASE 2: VERIFICATION")
    print(f"{'FICHIER':<20} | {'REJET %':<10} | {'RESULTAT'}")
    print("-" * 50)
    
    for path in files:
        y = preprocess_audio(path)
        if y is None: continue
            
        feats = extract_features(y)
        scores = gmm.score_samples(feats)
        
        # Combien de frames sont sous le seuil ?
        reject_count = np.sum(scores < score_threshold)
        rejection_rate = reject_count / len(scores)
        
        status = "🟢 ACCEPTE"
        # Si le taux de rejet dépasse TA limite + 5%, on coupe.
        if rejection_rate > max_reject_rate:
            status = "🔴 REJETTE"
            
        print(f"{path:<20} | {rejection_rate*100:>6.1f}%    | {status}")

# ==========================================
# EXECUTION
# ==========================================
train_files = ["data-wav\me1.wav", "data-wav\me2.wav"]
val_file = "data-wav\me3.wav"
test_list = ["data-wav\friend.wav", "data-wav\test_friend.wav", "data-wav\heey.wav"]

try:
    model, s_thresh, r_limit = train_system(train_files, val_file)
    test_files(model, s_thresh, r_limit, test_list)
except Exception as e:
    print(f"Erreur: {e}")