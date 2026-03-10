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
    dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train", streaming=False)

    # 4. Extraer el primer audio y guardarlo como temporal para XTTS
    ejemplo_audio = dataset_chileno[0]["audio"]
    ruta_temporal_ref = "./datasets/referencias/"+entrenamiento
    sf.write(ruta_temporal_ref, ejemplo_audio["array"], ejemplo_audio["sampling_rate"])

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
