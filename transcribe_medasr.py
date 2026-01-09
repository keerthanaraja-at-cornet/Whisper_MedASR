import os
from pathlib import Path
from transformers import pipeline
import torch
import librosa
import numpy as np
import re

# Hugging Face auth (expects HF_TOKEN in environment)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")
os.environ["HF_TOKEN"] = HF_TOKEN

# Paths
AUDIO_DIR = Path("audio")
OUTPUT_DIR = Path("transcriptions_medasr")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ID = "google/medasr"
print(f"Loading MedASR model: {MODEL_ID}")

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline(
    "automatic-speech-recognition",
    model=MODEL_ID,
    device=device,
)
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# Helpers

def denoise(audio: np.ndarray) -> np.ndarray:
    """Soft spectral gate to keep speech detail."""
    D = librosa.stft(audio)
    mag, phase = np.abs(D), np.angle(D)
    power_db = librosa.power_to_db(mag**2, ref=np.max(mag**2))
    thresh = np.percentile(power_db, 18)  # keep more low-energy speech
    mask = np.where(power_db > thresh, 1.0, 0.2)
    cleaned = mag * mask
    return librosa.istft(cleaned * np.exp(1j * phase))

def normalize_trim(audio: np.ndarray, top_db: int = 45) -> np.ndarray:
    max_val = np.max(np.abs(audio)) or 1.0
    audio = audio / max_val
    audio, _ = librosa.effects.trim(audio, top_db=top_db)
    return audio

def pre_emphasize(audio: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    return np.append(audio[0], audio[1:] - coeff * audio[:-1])

def rms_normalize(audio: np.ndarray, target_db: float = -18.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio**2)) or 1e-6
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)
    return audio * gain

REPLACEMENTS = [
    ("pulpitus", "pulpitis"),
    ("poulpitus", "pulpitis"),
    ("maggum", "amalgam"),
    ("magum", "amalgam"),
    ("concerur", "concord"),
    ("tosted", "toasted"),
    ("bagal", "bagel"),
    ("senapril", "lisinopril"),
    ("aren'", "aren't"),
    ("sli on", "slick on"),
    ("mark,", "Mark,"),
    ("mark.", "Mark."),
    ("doctor.", "Doctor."),
    ("Doctor Doctor", "Doctor"),
    ("mark yes", "Mark: Yes"),
    ("mark no", "Mark: No"),
    ("mark right", "Mark: Right"),
    ("pain was the same", "pain was insane"),
    ("delta dental", "Delta Dental"),
    ("6mm", "6 mm"),
    ("number four", "number 4"),
    ("15 seconds", "15 seconds"),
]

def clean_text(text: str) -> str:
    for wrong, right in REPLACEMENTS:
        text = text.replace(wrong, right)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("..", ".").replace(",,", ",")
    return text

# Collect audio files
audio_files = [f for f in AUDIO_DIR.iterdir() if f.suffix.lower() in {".mp3",".wav",".m4a",".flac",".ogg",".mp4"}]
if not audio_files:
    print("No audio files found in the audio folder!")
    raise SystemExit(1)

print(f"Found {len(audio_files)} audio file(s). Starting MedASR transcription...\n")

for audio_path in audio_files:
    out_path = OUTPUT_DIR / f"{audio_path.stem}_medasr_transcript.txt"
    print(f"Transcribing: {audio_path.name}")
    print(f"Output: {out_path}")

    try:
        # Load
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        # Enhance
        audio = denoise(audio)
        audio = normalize_trim(audio, top_db=45)
        audio = rms_normalize(audio, target_db=-18.0)
        audio = pre_emphasize(audio, coeff=0.97)

        # Transcribe with larger chunk and modest overlap for higher accuracy
        result = pipe(
            {"array": audio, "sampling_rate": sr},
            chunk_length_s=16,
            stride_length_s=1,
        )

        transcript = clean_text(result["text"])

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✓ Saved ({len(transcript)} chars)")
        print(f"Preview: {transcript[:220]}...")
        print("-" * 80)

    except Exception as exc:
        print(f"✗ Error on {audio_path.name}: {exc}")
        import traceback
        traceback.print_exc()
        print("-" * 80)

print("✓ MedASR transcription complete! Check transcriptions_medasr.")
