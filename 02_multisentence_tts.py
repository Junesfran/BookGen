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


# A MEDIO HACER
# ----------------------------------------------------------------
recorer_json = {
  "escenas": [
    {
      "personaje": "Bernarda",
      "texto": "¡Silencio! ¿Es que no se puede estar en esta casa con un poco de paz?",
      "emocion": "ira"
    },
    {
      "personaje": "Adela",
      "texto": "¡Yo no aguantaré más tus gritos! Me iré de aquí, aunque sea al arroyo.",
      "emocion": "desafiante"
    },
    {
      "personaje": "La Poncia",
      "texto": "Bernarda, deja a la muchacha. La sangre se le sube a la cabeza con este calor que hace.",
      "emocion": "calma"
    }
  ]
}

escena = recorer_json["escenas"]
voces = len(embeddings_dataset)

personajes = {}

for x in escena:
    rn = np.random.randint(0, voces)
    personajes[x['personaje']] = rn

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

for i, x in enumerate(escena):
    personaje = x["personaje"]
    texto = x['texto']
    voz = personajes[personaje]

    print(f"\n--- Escena {i+1} ---")
    print(f"{personaje}: {texto}")

    # Prepare speaker embedding for this voice
    speaker_embedding = torch.tensor(
        embeddings_dataset[voz]["speaker_embeddings"]
    ).unsqueeze(0)

    # Tokenize and generate
    inputs = processor(text=texto, return_tensors="pt")

    with torch.no_grad():
        speech = model.generate_speech(
            inputs["input_ids"],
            speaker_embedding,
            vocoder=vocoder
        )

    duration = speech.shape[0] / SAMPLE_RATE
    print(f"Generated {speech.shape[0]} samples ({duration:.2f}s)")

    # Save individual file
    # output_path = os.path.join(OUTPUT_DIR, f"output_sentence_{i + 1}.wav")
    speech_numpy = speech.numpy()
    # write_wav(output_path, rate=SAMPLE_RATE, data=(speech_numpy * 32767).astype(np.int16))
    # print(f"Saved: {output_path}")

    all_waveforms.append(speech_numpy)

# ----------------------------------------------------------------
# 4. Concatenate all waveforms with short silence gaps
# ----------------------------------------------------------------
silence_frases = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)  # 0.5s silence
silence_personajes = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)  # 0.8s silence

combined_parts = []
for i, wf in enumerate(all_waveforms):
    combined_parts.append(wf)

    if i < len(all_waveforms) - 1:
        actual = escena[i]["personaje"]
        siguiente = escena[i+1]["personaje"]

        if actual != siguiente:
            combined_parts.append(silence_personajes)
        else:
            combined_parts.append(silence_frases)

combined = np.concatenate(combined_parts)
combined_path = os.path.join(OUTPUT_DIR, "output_combined.wav")
write_wav(combined_path, rate=SAMPLE_RATE, data=(combined * 32767).astype(np.int16))

total_duration = combined.shape[0] / SAMPLE_RATE
print(f"\nCombined audio: {total_duration:.2f}s saved to {combined_path}")
print("Play it with: afplay output_combined.wav")