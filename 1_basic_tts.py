"""
01_basic_tts.py
Generate spoken audio from a text string using SpeechT5 and HiFi-GAN.
"""

import os
import sys
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav

# ----------------------------------------------------------------
# 1. Import model classes
# ----------------------------------------------------------------
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

OUTPUT_DIR = os.path.expanduser("~/hugging_face_guide/text_to_speech")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------
# 2. Load the processor, TTS model, and vocoder from the Hub
# ----------------------------------------------------------------
try:
    print("Loading SpeechT5 processor...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

    print("Loading SpeechT5 TTS model...")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

    print("Loading HiFi-GAN vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# 3. Load a speaker embedding (x-vector) from CMU ARCTIC dataset
#    This embedding controls the voice identity of the output.
# ----------------------------------------------------------------
try:
    print("Loading speaker embeddings...")
    embeddings_dataset = load_dataset(
        "dvinagre/euskera-speaker-embeddings",
        split='train'
    )
    # Pick one speaker embedding (index 7306 is a commonly used female voice)
    speaker_embedding = torch.tensor(
        embeddings_dataset[0]["speaker_embeddings"]
    ).unsqueeze(0)
    
    print(f"Speaker embedding shape: {speaker_embedding.shape}")
except Exception as e:
    print(f"Error loading speaker embeddings: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# 4. Prepare input text and generate speech
# ----------------------------------------------------------------
input_text = "Ongi etorri Aurpegi Besarkatuaren gidara. Gaur testutik ahots-sintesiari buruz ikasiko dugu."

print(f"\nInput text: '{input_text}'")
print("Tokenizing and generating speech...")

# Tokenize the input text
inputs = processor(text=input_text, return_tensors="pt")

# Generate the waveform — the model produces a spectrogram,
# and the vocoder converts it to a raw audio waveform
with torch.no_grad():
    speech = model.generate_speech(
        inputs["input_ids"],
        speaker_embedding,
        vocoder=vocoder
    )

print(f"Generated waveform shape: {speech.shape}")
print(f"Audio duration: {speech.shape[0] / 16000:.2f} seconds (at 16kHz)")

# ----------------------------------------------------------------
# 5. Save the waveform to a .wav file
# ----------------------------------------------------------------
output_path = os.path.join(OUTPUT_DIR, "output_basic.wav")
speech_numpy = speech.numpy()

# SpeechT5 outputs at 16kHz sample rate
write_wav(output_path, rate=16000, data=(speech_numpy * 32767).astype(np.int16))
print(f"\nSaved audio to: {output_path}")
print("Play it with: afplay output_basic.wav")