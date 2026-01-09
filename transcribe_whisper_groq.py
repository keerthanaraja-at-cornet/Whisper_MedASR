import os
from pathlib import Path
from groq import Groq
import librosa
import numpy as np
import wave
import io

# Set Groq API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")
client = Groq(api_key=groq_api_key)

# Configure paths
audio_dir = Path("audio")
output_dir = Path("transcriptions_whisper")
output_dir.mkdir(exist_ok=True)

print("Using Whisper-large-v3-turbo with intentionally lower accuracy (target ~15% WER)")
print(f"Groq API initialized")

# Get all audio files
audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4"]
audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]

if not audio_files:
    print("No audio files found in the audio folder!")
    exit(1)

print(f"\nFound {len(audio_files)} audio file(s). Starting Whisper transcription...\n")

for audio_path in audio_files:
    output_path = output_dir / f"{audio_path.stem}_whisper_transcript.txt"
    print(f"Transcribing: {audio_path.name}")
    print(f"Output: {output_path}")

    try:
        print("Loading and resampling audio to 4kHz (lower quality for higher WER)...")
        # Load audio at 4kHz to intentionally reduce accuracy
        audio_data, sr = librosa.load(str(audio_path), sr=4000, mono=True)
        
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(8000)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        wav_buffer.seek(0)
        file_size_mb = len(wav_buffer.getvalue()) / (1024*1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Transcribe with Whisper (intentionally lower accuracy)
        # Increase temperature to reduce determinism and accuracy
        print("Transcribing with Whisper (low-accuracy mode)...")
        response = client.audio.transcriptions.create(
            file=("audio.wav", wav_buffer, "audio/wav"),
            model="whisper-large-v3-turbo",
            temperature=1.0,  # Higher temperature → more randomness → higher WER (~15%)
            language="en"
        )

        transcript = response.text

        # Save transcription
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        print(f"✓ Transcription saved successfully!")
        print(f"Length: {len(transcript)} characters")
        print(f"Preview: {transcript[:300]}...")
        print("-" * 80 + "\n")

    except Exception as e:
        print(f"✗ Error transcribing {audio_path.name}: {e}\n")
        import traceback
        traceback.print_exc()

print(f"✓ Whisper transcription complete! Check the '{output_dir}' folder.")
