import numpy as np
import librosa
from sklearn.mixture import GaussianMixture

# -----------------------------
# 1. Extraire MFCC d'un audio
# -----------------------------
def extract_mfcc(path, n_mfcc=20):
    audio, sr = librosa.load(path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # shape (time_steps, features)

# -----------------------------
# 2. Entraîner un GMM sur plusieurs audios
# -----------------------------
def train_gmm_multiple(audio_paths, n_components=8):
    mfcc_list = []

    for path in audio_paths:
        mfcc = extract_mfcc(path)
        mfcc_list.append(mfcc)

    # Combine tous les MFCC en un seul tableau
    mfcc_all = np.vstack(mfcc_list)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',
        max_iter=200
    )

    gmm.fit(mfcc_all)
    return gmm

# -----------------------------
# 3. Tester si un audio correspond à la même personne
# -----------------------------
def is_same_speaker(gmm_model, test_audio_path, threshold=-200):
    mfcc_test = extract_mfcc(test_audio_path)
    score = gmm_model.score(mfcc_test)

    print(f"Log-likelihood score : {score}")

    return score > threshold

# -----------------------------
# Exemple d'utilisation
# -----------------------------
reference_audios = [
    "audio1.wav",
    "audio2.wav",
    "audio3.wav"
]

print("📢 Entraînement du modèle GMM...")
gmm = train_gmm_multiple(reference_audios)

print("🔍 Test...")
result = is_same_speaker(gmm, "test_voice.wav")

if result:
    print("✔ Même personne")
else:
    print("❌ Personne différente")
