import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import pandas as pd
from datasets import load_dataset
import torch
import torchaudio
import io
from speechbrain.inference.speaker import EncoderClassifier

def descargar_modelos():
    # Descargar
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    # Guardar en carpetas locales
    processor.save_pretrained("./models/speecht5_tts")
    model.save_pretrained("./models/speecht5_tts")
    vocoder.save_pretrained("./models/speecht5_hifigan")

def descargar_chileno():
    dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train")
    df = dataset_chileno.to_pandas()
    df['audio'].head()

    # 1. Cargamos el modelo (512 dimensiones)
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir="pretrained_models/xvect"
    )

    def generar_xvector_desde_bytes(elemento_audio):
        """
        elemento_audio: es el diccionario que ves en tu df, 
        ej: {'bytes': b'RIFF...', 'path': '...'}
        """
        # Extraer los bytes del diccionario
        audio_bytes = elemento_audio['bytes']
        
        # Convertir bytes a un objeto que torchaudio pueda leer
        buffer = io.BytesIO(audio_bytes)
        
        # Cargar el audio desde el buffer
        signal, sr = torchaudio.load(buffer)
        
        # --- Pre-procesado estándar ---
        # 1. Convertir a 16kHz si es distinto
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)
            
        # 2. Convertir a mono si es estéreo
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        with torch.no_grad():
            # Extraer el embedding [1, 512]
            embeddings = classifier.encode_batch(signal)
            xvector = embeddings.squeeze(1)
            
        return xvector

    # --- EJECUCIÓN ---

    # 1. Tomamos el primer elemento de tu df
    audio_data = df['audio'][0]

    # 2. Generamos el vector
    xvector_tensor = generar_xvector_desde_bytes(audio_data)

    # 3. Guardar y descargar
    xvector_np = xvector_tensor.cpu().numpy()
    nombre_archivo = "./datasets/chileno.npy"
    np.save(nombre_archivo, xvector_np)

    print(f"✅ ¡Conseguido! X-vector extraído de los bytes y listo para descargar.")    