import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

# Audio parameters
SR = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 128
TARGET_DURATION = 5.0
TARGET_LENGTH = int(TARGET_DURATION * SR)

def preprocess_audio(file_path):
    # Load audio (force mono)
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Normalize amplitude
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak * 0.99

    # Resample
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Split audio into 5s chunks
    chunks = []
    for start in range(0, len(y), TARGET_LENGTH):
        chunk = y[start:start + TARGET_LENGTH]
        if len(chunk) < TARGET_LENGTH:
            chunk = np.pad(chunk, (0, TARGET_LENGTH - len(chunk)), mode="constant")

        # Convert to Mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=chunk, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Convert spectrogram to RGBA image
        fig = plt.figure(figsize=(3, 3))
        librosa.display.specshow(S_dB, sr=SR, hop_length=HOP_LENGTH, cmap="magma")
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        buf.seek(0)
        img = Image.open(buf).convert("RGBA")  # 4 channels
        chunks.append(img)

    return chunks