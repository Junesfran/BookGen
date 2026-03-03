"""
02_multi_sentence_tts.py
Generate speech for multiple sentences with different speaker voices.
"""

import os
import sys
import torch
import numpy as np
from scipy.io.wavfile import write as write_wav

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

OUTPUT_DIR = os.path.expanduser("~/hugging_face_guide/text_to_speech")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 16000

# ----------------------------------------------------------------
# 1. Load models and speaker embeddings
# ----------------------------------------------------------------
try:
    processor = SpeechT5Processor.from_pretrained(
        "./models/speecht5_tts",
        local_files_only=True
        )
    
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "./models/speecht5_tts",
        local_files_only=True
        )
    
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan",
        local_files_only=True
        )
    
    embeddings_dataset = load_dataset(
        'parquet',
        data_files="./datasets/euskera.parquet",
        split='train'
    )
    
    print(f"Loaded {len(embeddings_dataset)} speaker embeddings.")
except Exception as e:
    print(f"Error during model/data loading: {e}")
    sys.exit(1)

# ----------------------------------------------------------------
# 2. Define sentences and speaker indices
#    Different indices produce different voice characteristics.
# ----------------------------------------------------------------
sentences = [
    "Hugging Face makes natural language processing accessible to everyone.",
    "Text to speech models can read your documents aloud.",
    "Neural vocoders produce high quality audio waveforms.",
]

# Three different speaker embedding indices for variety
speaker_indices = [0, 2000, 5000]

# ----------------------------------------------------------------
# 3. Generate audio for each sentence
# ----------------------------------------------------------------
all_waveforms = []

for i, (text, spk_idx) in enumerate(zip(sentences, speaker_indices)):
    print(f"\n--- Sentence {i + 1} (speaker index {spk_idx}) ---")
    print(f"Text: '{text}'")

    # Prepare speaker embedding for this voice
    speaker_embedding = torch.tensor(
        embeddings_dataset[spk_idx]["speaker_embeddings"]
    ).unsqueeze(0)

    # Tokenize and generate
    inputs = processor(text=text, return_tensors="pt")

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embedding,
            vocoder=vocoder
        )

    duration = speech.shape[0] / SAMPLE_RATE
    print(f"Generated {speech.shape[0]} samples ({duration:.2f}s)")

    # Save individual file
    output_path = os.path.join(OUTPUT_DIR, f"output_sentence_{i + 1}.wav")
    speech_numpy = speech.numpy()
    write_wav(output_path, rate=SAMPLE_RATE, data=(speech_numpy * 32767).astype(np.int16))
    print(f"Saved: {output_path}")

    all_waveforms.append(speech_numpy)

# ----------------------------------------------------------------
# 4. Concatenate all waveforms with short silence gaps
# ----------------------------------------------------------------
silence = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)  # 0.5s silence
combined_parts = []
for j, wf in enumerate(all_waveforms):
    combined_parts.append(wf)
    if j < len(all_waveforms) - 1:
        combined_parts.append(silence)

combined = np.concatenate(combined_parts)
combined_path = os.path.join(OUTPUT_DIR, "output_combined.wav")
write_wav(combined_path, rate=SAMPLE_RATE, data=(combined * 32767).astype(np.int16))

total_duration = combined.shape[0] / SAMPLE_RATE
print(f"\nCombined audio: {total_duration:.2f}s saved to {combined_path}")
print("Play it with: afplay output_combined.wav")