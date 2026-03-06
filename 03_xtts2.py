import torch
from TTS.api import TTS
import os

# 1. Configurar el dispositivo (Usa GPU si tienes, si no CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Cargar el modelo XTTS v2 (Es multilingüe y clona voces)
# La primera vez tardará en descargar, es normal.
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 3. Preparar la carpeta de salida
os.makedirs("./output", exist_ok=True)

# 4. Datos para la generación
texto_a_decir = "Hola, estoy probando una clonación de voz mucho más natural que el modelo anterior."
# Aquí pon la ruta a un audio real de tu dataset (un archivo .wav de unos 6-10 segundos)
dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train")
df = dataset_chileno.to_pandas()
ruta_audio_referencia = df['audio'][0]

# 5. Generación directa a archivo
tts.tts_to_file(
    text=texto_a_decir,
    speaker_wav=ruta_audio_referencia,
    language="es",
    file_path="./output/resultado_natural.wav"
)

print("¡Audio generado en ./output/resultado_natural.wav!")