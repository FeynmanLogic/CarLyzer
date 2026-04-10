# 🎧 Cognitive Enrichment Score (CES) Analyzer

## 📌 Overview

Psychoacoustics is a multi-disciplinary field that combines **Digital Signal Processing (DSP)**, **Computer Science**, **Physics**, **Cognitive Science**, and **Biology** among others to see how the human ear reacts to sound.
And yes, I did make a fair use of AI while building this.

This project aims to build a tool that computes a **Cognitive Enrichment Score (CES)** for a given song using its **YouTube link**. The goal is to provide a quantitative proxy for how cognitively engaging or enriching a piece of music may be, helping users make informed listening choices.

---
Note: > ⚠️ FFmpeg is required for audio extraction. Make sure it is installed and added to PATH, or specify its location in the code.
## 🧠 What is Cognitive Enrichment Score (CES)?

CES is a **computed proxy metric** that estimates how engaging a piece of audio is for the human brain, based on psychoacoustic principles.

It is derived from features such as:

* **Spectral Entropy** → richness of sound
* **Pitch Variability** → melodic complexity (e.g., gamakas in Carnatic music)
* **Temporal Structure** → rhythm and timing variation
* **Harmonic Ratio** → structured vs noisy components

> ⚠️ Note: CES is not a direct measure of cognition, but a **signal-based approximation** grounded in psychoacoustic theory.

---

## 🚀 Features

* 🎥 Input: YouTube link only
* 🔊 Automatic audio extraction
* 🎼 Feature extraction using `librosa`
* 🧠 CES computation using psychoacoustic heuristics
* 🤖 Neural network-based refinement (PyTorch)
* 📊 Final cognitive enrichment score output

---

## ⚙️ Installation

```bash
pip install yt-dlp librosa torch numpy scipy
```

> Ensure you have **FFmpeg** installed for audio processing.

---

## ▶️ Usage

Run the script:

```bash
python main.py
```

Then input a YouTube link:

```bash
Enter YouTube URL: https://youtube.com/...
```

Output:

```
CES (proxy): 0.8421
NN score: 0.5123
Final Cognitive Enrichment Score: 0.7435
```

---

## 🏗️ Architecture

```
YouTube Link
     ↓
Audio Download (yt-dlp)
     ↓
Feature Extraction (librosa)
     ↓
CES Proxy Calculation
     ↓
Neural Network (PyTorch)
     ↓
Final Score
```

---

## 📊 Methodology

### Feature extraction:
Using `librosa`, we extract:

* Spectral Entropy → information richness of the signal  
* Spectral Centroid Variability → proxy for pitch variation  
* Harmonic Ratio → tonal vs noisy structure  
* Tempo → rhythmic structure

### 2. CES Proxy

A weighted combination:

* 30% Spectral Entropy
* 30% Pitch Variability
* 20% Harmonic Ratio
* 20% Tempo (normalized)
CES is computed on a short audio segment (typically first 30–60 seconds),
based on the assumption that psychoacoustic characteristics stabilize over time.

### 3. Neural Network

* Feedforward architecture
* Learns mapping from features → CES
* Currently untrained (acts as placeholder for future learning)

---

## ⚠️ Limitations

* CES is a **proxy**, not ground truth
* Neural network is **not trained by default**
* Cultural and subjective factors are not modeled
* YouTube compression may affect audio quality

---

## 🔬 Future Work

* Train neural network on:

  * Human preference data
  * EEG / attention datasets

* Upgrade model:

  * CNN on spectrograms
  * Transformer-based audio models

* Comparative studies:

  * Carnatic vs Western vs Pop
  * Artist-wise cognitive profiles

* Real-time recommendation system

---

## 💡 Research Potential

This project sits at the intersection of:

* Psychoacoustics
* Machine Learning
* Music Information Retrieval (MIR)

With proper validation, it can evolve into a **publishable research framework** for quantifying cognitive engagement in music.

---

## 📜 License

MIT License (or choose your preferred license)

---

## 🙌 Acknowledgements

* `librosa` for audio analysis
* `yt-dlp` for YouTube extraction
* `PyTorch` for deep learning

---
