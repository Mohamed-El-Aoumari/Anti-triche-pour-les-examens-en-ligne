import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
import soundfile as sf

# -----------------------------
# 1. Extraire les MFCC d'un audio
# -----------------------------
def extract_mfcc(path, n_mfcc=20):
    audio, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape (time, features)

# -----------------------------
# 2. Entraîner un GMM sur la voix de référence
# -----------------------------
def train_gmm(audio_reference_path, n_components=4):
    mfcc = extract_mfcc(audio_reference_path)
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200)
    gmm.fit(mfcc)
    return gmm

# -----------------------------
# 3. Tester si un autre audio provient de la même personne
# -----------------------------
def is_same_speaker(gmm_model, test_audio_path, threshold=-200):
    mfcc_test = extract_mfcc(test_audio_path)
    score = gmm_model.score(mfcc_test)  # log-likelihood moyenne

    print(f"Log-likelihood score : {score}")

    if score > threshold:
        return True  # même personne
    else:
        return False  # autre personne

# -----------------------------
# 4. Exemple d'utilisation
# -----------------------------
ref_audio = "reference.wav"        # audio de la personne principale
test_audio = "test.wav"            # audio à tester

print("📢 Entraînement du modèle GMM...")
gmm = train_gmm(ref_audio)

print("🔍 Test d'identité...")
result = is_same_speaker(gmm, test_audio)

if result:
    print("✔ Même personne")
else:
    print("❌ Personne différente")
