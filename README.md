# Whisper vs MedASR Comparison

A comprehensive comparison study between OpenAI's Whisper and Google's MedASR for medical speech recognition.

## Overview

This project evaluates the performance of two state-of-the-art automatic speech recognition (ASR) models on medical conversations:
- **Whisper (large-v3-turbo)** via Groq API
- **MedASR** by Google (specialized for medical terminology)

## Features

- 🎙️ Audio transcription using both Whisper and MedASR
- 📊 Comprehensive metrics calculation (WER, accuracy, precision, etc.)
- 📝 Detailed technical report in Jupyter Notebook format
- 🔍 Side-by-side comparison of transcription quality

## Project Structure

```
├── audio/                          # Input audio files (not included in repo)
├── transcriptions_whisper/         # Whisper transcription outputs
├── transcriptions_medasr/          # MedASR transcription outputs
├── metrics/
│   ├── calculate_metrics.py        # Metrics calculation script
│   └── comprehensive_metrics_results.txt
├── conversations/
│   └── convo.txt                   # Reference conversation text
├── transcribe_whisper_groq.py      # Whisper transcription script
├── transcribe_medasr.py            # MedASR transcription script
└── Technical_Report_Whisper_vs_MedASR.ipynb  # Analysis notebook
```

## Setup

### Prerequisites

- Python 3.8+
- Groq API key
- Hugging Face token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/keerthanaraja-at-cornet/Whisper_MedASR.git
cd Whisper_MedASR
```

2. Install dependencies:
```bash
pip install groq transformers torch librosa numpy jiwer
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# GROQ_API_KEY=your_groq_api_key_here
# HF_TOKEN=your_huggingface_token_here
```

## Usage

### Transcribe with Whisper

```bash
python transcribe_whisper_groq.py
```

### Transcribe with MedASR

```bash
python transcribe_medasr.py
```

### Calculate Metrics

```bash
python metrics/calculate_metrics.py
```

## Results

See the [Technical Report](Technical_Report_Whisper_vs_MedASR.ipynb) for detailed analysis and comparison results.

## Key Metrics Evaluated

- Word Error Rate (WER)
- Character Error Rate (CER)
- Precision, Recall, F1-Score
- Processing time
- Medical terminology accuracy

## License

This project is for educational and research purposes.

## Acknowledgments

- OpenAI Whisper
- Google MedASR
- Groq for API access