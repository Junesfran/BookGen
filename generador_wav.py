import torch
from TTS.api import TTS
import os
from datasets import load_dataset
import soundfile as sf # Necesario para guardar el audio de referencia


def generar_wav(entrenamiento: str):
    # 1. Definir el dispositivo (Cuda si tienes Nvidia, si no CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # 2.1 Cargar el modelo (Esto descargará ~2GB la primera vez)
    # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # 2.2 En caso de tener el modelo en local¡
    tts = TTS(
        model_path="./models/xtts_v2",
        config_path="./models/xtts_v2/config.json"
        ).to(device)

    os.makedirs("./output", exist_ok=True)

    # 3. Cargar el dataset chileno
    print("Cargando dataset...")
    dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train", streaming=False)

    # 4. Extraer el primer audio y guardarlo como temporal para XTTS
    # XTTS necesita un archivo físico, no un objeto de memoria
    ejemplo_audio = dataset_chileno[0]["audio"]
    ruta_temporal_ref = "./datasets/referencias/"+entrenamiento
    sf.write(ruta_temporal_ref, ejemplo_audio["array"], ejemplo_audio["sampling_rate"])

    archivo = open('./datasets/Don_Quijote.txt', 'r')
    texto_a_decir = archivo.read()

    # 5. Generar el audio
    print("Generando voz... ten paciencia, XTTS es pesado.")
    tts.tts_to_file(
        text=texto_a_decir,
        speaker_wav=ruta_temporal_ref, # Ahora sí es una ruta válida
        #speaker_wav='clonacion_voz.wav',
        language="es",
        file_path="./output/resultado.wav"
    )

    print("¡Éxito! Audio generado en ./output/resultado.wav")
