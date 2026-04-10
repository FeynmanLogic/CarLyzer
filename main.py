# ==============================
# Cognitive Enrichment Analyzer
# ==============================

import yt_dlp
import librosa
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import entropy
import os

# ------------------------------
# 1. Download audio from YouTube
# ------------------------------
def download_audio(url, filename="audio.wav"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'quiet': True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    return "temp.wav"


# ------------------------------
# 2. Feature Extraction
# ------------------------------
def extract_features(file):
    y, sr = librosa.load(file, sr=22050)

    # --- Spectral Entropy ---
    S = np.abs(librosa.stft(y))
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-6)
    spectral_entropy = np.mean([entropy(frame) for frame in S_norm.T])

    # --- Pitch Variability ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_vals = pitches[magnitudes > np.median(magnitudes)]
    pitch_var = np.std(pitch_vals) if len(pitch_vals) > 0 else 0

    # --- Tempo ---
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # --- Harmonic Ratio ---
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-6)

    # --- Mel Spectrogram (compressed) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)
    mel_mean = np.mean(mel_db, axis=1)

    features = np.concatenate([
        mel_mean,
        [spectral_entropy, pitch_var, tempo, harmonic_ratio]
    ])

    return features


# ------------------------------
# 3. CES Proxy (baseline score)
# ------------------------------
def compute_ces_proxy(features):
    spectral_entropy = features[-4]
    pitch_var = features[-3]
    tempo = features[-2]
    harmonic_ratio = features[-1]

    CES = (
        0.3 * spectral_entropy +
        0.3 * pitch_var +
        0.2 * harmonic_ratio +
        0.2 * (tempo / 200.0)  # normalize tempo
    )

    return CES


# ------------------------------
# 4. Neural Network Model
# ------------------------------
class CESModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# 5. Main Pipeline
# ------------------------------
def analyze_youtube(url):
    print("\nDownloading audio...")
    file = download_audio(url)

    print("Extracting features...")
    features = extract_features(file)

    print("Computing baseline CES...")
    ces_proxy = compute_ces_proxy(features)

    # Prepare NN
    model = CESModel(len(features))

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    # Forward pass (no training yet)
    nn_score = model(x).item()

    final_score = 0.7 * ces_proxy + 0.3 * nn_score

    # Cleanup
    if os.path.exists(file):
        os.remove(file)

    return {
        "CES_proxy": ces_proxy,
        "NN_score": nn_score,
        "Final_score": final_score
    }


# ------------------------------
# 6. Run
# ------------------------------
if __name__ == "__main__":
    url = input("Enter YouTube URL: ")

    results = analyze_youtube(url)

    print("\n==== Results ====")
    print(f"CES (proxy): {results['CES_proxy']:.4f}")
    print(f"NN score: {results['NN_score']:.4f}")
    print(f"Final Cognitive Enrichment Score: {results['Final_score']:.4f}")
