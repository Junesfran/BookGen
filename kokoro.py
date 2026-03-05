"""
02_multi_sentence_tts_kokoro.py
Generate speech for multiple sentences using Kokoro TTS.
"""

import os
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUTPUT_DIR = os.path.expanduser("~/hugging_face_guide/text_to_speech")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 24000

# ------------------------------------------------------------
# 1. Load Kokoro pipeline
# ------------------------------------------------------------

pipeline = KPipeline(lang_code='a')

# Available voices (example)
VOICES = [
    "af_heart",
    "af_bella",
    "am_michael",
    "am_adam"
]

# ------------------------------------------------------------
# 2. Scene definition
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 3. Assign random voice per character
# ------------------------------------------------------------

personajes = {}

for x in escena:
    if x["personaje"] not in personajes:
        personajes[x["personaje"]] = np.random.choice(VOICES)

# ------------------------------------------------------------
# 4. Generate audio
# ------------------------------------------------------------

all_waveforms = []

for i, x in enumerate(escena):

    personaje = x["personaje"]
    texto = x["texto"]
    voz = personajes[personaje]

    print(f"\n--- Escena {i+1} ---")
    print(f"{personaje} ({voz}): {texto}")

    generator = pipeline(texto, voice=voz)

    for gs, ps, audio in generator:
        all_waveforms.append(audio)

# ------------------------------------------------------------
# 5. Concatenate with silence
# ------------------------------------------------------------

silence_frases = np.zeros(int(0.5 * SAMPLE_RATE))
silence_personajes = np.zeros(int(0.8 * SAMPLE_RATE))

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

combined_path = os.path.join(OUTPUT_DIR, "output_kokoro.wav")

sf.write(combined_path, combined, SAMPLE_RATE)

print(f"\nSaved to {combined_path}")