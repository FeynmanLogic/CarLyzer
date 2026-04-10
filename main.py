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
    'ffmpeg_location': r'C:\Users\USER\ffmpeg-2026-04-09-git-d3d0b7a5ee-full_build\bin', #replace this with the file location of the bin directory of the ffmpeg in ur system
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
    'quiet': True
}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    title = info.get("title", "unknown")

    return "temp.wav", title
import csv
import os

def log_results(title, ces, nn_score, final_score, filename="results.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "title",
                "ces_proxy",
                "nn_score",
                "final_score"
            ])

        writer.writerow([
            title,
            round(ces, 4),
            round(nn_score, 4),
            round(final_score, 4)
        ])

# ------------------------------
# 2. Feature Extraction
# ------------------------------
from tqdm import tqdm
import librosa
import numpy as np
from scipy.stats import entropy

def extract_features(file):
    print("\nLoading audio (first 60 sec)...")
    y, sr = librosa.load(file, sr=16000, duration=60) #change duration to whatever need be

    results = []

    steps = [
        "Spectral Entropy",
        "Pitch Variability (Centroid)",
        "Harmonic Ratio",
        "Tempo"
    ]

    for step in tqdm(steps, desc="Feature Extraction", ncols=80):

        if step == "Spectral Entropy":
            S = np.abs(librosa.stft(y, n_fft=1024))
            S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-6)
            spectral_entropy = np.mean([entropy(frame) for frame in S_norm.T])
            results.append(float(spectral_entropy))  # ✅ scalar

        elif step == "Pitch Variability (Centroid)":
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            pitch_var = np.std(centroid)
            results.append(float(pitch_var))  # ✅ scalar

        elif step == "Harmonic Ratio":
            y_harmonic, _ = librosa.effects.hpss(y)
            harmonic_ratio = np.sum(y_harmonic**2) / (np.sum(y**2) + 1e-6)
            results.append(float(harmonic_ratio))  # ✅ scalar

        elif step == "Tempo":
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            results.append(float(tempo))  # ✅ scalar

    # ✅ Convert to clean numeric array
    features = np.array(results, dtype=np.float32)

    # 🧪 DEBUG CHECKS
    print("\n--- DEBUG: Feature Vector ---")
    print("Features:", features)
    print("Shape:", features.shape)
    print("Dtype:", features.dtype)

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
    file, title = download_audio(url)
    print(f"\n🎵 Song: {title}")

    print("Extracting features...")
    features = extract_features(file)

    print("Computing baseline CES...")
    ces_proxy = compute_ces_proxy(features)

    # Prepare NN
    model = CESModel(len(features))

    # Convert to tensor
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    model.eval()  # 🔥 important fix
    with torch.no_grad():
     nn_score = model(x).item()

    final_score = 0.7 * ces_proxy + 0.3 * nn_score

    # Cleanup
    if os.path.exists(file):
        os.remove(file)
    log_results(title, ces_proxy, nn_score, final_score)

    return {
        "CES_proxy": ces_proxy,
        "NN_score": nn_score,
        "Final_score": final_score
    }
import csv
import os

def save_to_dataset(features, ces, filename="dataset.csv"):
    file_exists = os.path.isfile(filename)

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "spectral_entropy",
                "pitch_var",
                "harmonic_ratio",
                "tempo",
                "ces"
            ])

        writer.writerow(list(features) + [ces])

# ------------------------------
# 6. Run
# ------------------------------
if __name__ == "__main__":
    url = input("Enter YouTube URL: ")

    results = analyze_youtube(url)
    save_to_dataset(features, ces_proxy)

    print("\n==== Results ====")
    print(f"CES (proxy): {results['CES_proxy']:.4f}")
    print(f"NN score: {results['NN_score']:.4f}")
    print(f"Final Cognitive Enrichment Score: {results['Final_score']:.4f}")
