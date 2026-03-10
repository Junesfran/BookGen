import torch
from TTS.api import TTS
import os
from datasets import load_dataset
import soundfile as sf


def generar_wav(entrenamiento: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")


    # 2. Cargar el modelo en local
    tts = TTS(
        model_path="./models/xtts_v2",
        config_path="./models/xtts_v2/config.json"
        ).to(device)

    os.makedirs("./output", exist_ok=True)

    # 3. Cargar el dataset chileno
    print("Cargando dataset...")
    
    # 4. Extraer el primer audio y guardarlo como temporal para XTTS
    ruta_temporal_ref = "./datasets/referencias/"+entrenamiento

    # 4.4. Usaremos como muestra de ejemplo un fragmento del Quijote
    archivo = open('./datasets/Don_Quijote.txt', 'r')
    texto_a_decir = archivo.read()

    # 5. Generar el audio
    print("Generando voz... ten paciencia, XTTS es pesado.")
    tts.tts_to_file(
        text=texto_a_decir,
        speaker_wav=ruta_temporal_ref, 
        language="es",
        file_path="./output/resultado.wav"
    )

    print("¡Éxito! Audio generado en ./output/resultado.wav")
