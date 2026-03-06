import torch
from TTS.api import TTS
import os
from datasets import load_dataset
import soundfile as sf # Necesario para guardar el audio de referencia

# 1. Definir el dispositivo (Cuda si tienes Nvidia, si no CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# 2. Cargar el modelo (Esto descargará ~2GB la primera vez)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

os.makedirs("./output", exist_ok=True)

# 3. Cargar el dataset chileno
print("Cargando dataset...")
dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train", streaming=False)

# 4. Extraer el primer audio y guardarlo como temporal para XTTS
# XTTS necesita un archivo físico, no un objeto de memoria
ejemplo_audio = dataset_chileno[0]["audio"]
ruta_temporal_ref = "referencia_chilena.wav"
sf.write(ruta_temporal_ref, ejemplo_audio["array"], ejemplo_audio["sampling_rate"])

texto_a_decir = "Hola, cachai que ahora estoy probando una clonación con acento chileno mucho más natural."

# 5. Generar el audio
print("Generando voz... ten paciencia, XTTS es pesado.")
tts.tts_to_file(
    text=texto_a_decir,
    speaker_wav=ruta_temporal_ref, # Ahora sí es una ruta válida
    language="es",
    file_path="./output/resultado_chileno_natural.wav"
)

print("¡Éxito! Audio generado en ./output/resultado_chileno_natural.wav")