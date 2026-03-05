import torch
import torchaudio
import numpy as np
from datasets import load_dataset
from speechbrain.pretrained import EncoderClassifier  # versión nueva

# Cargar dataset
dataset = load_dataset("ylacombe/google-chilean-spanish", "female", split="train")

# Cargar modelo (512 dims)
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="pretrained_models/xvect"
)

def generar_xvector(audio_dict):
    """
    audio_dict: {'array': np.array, 'sampling_rate': int}
    Compatible con torchaudio >=2.6
    """
    waveform = torch.tensor(audio_dict["array"]).float()

    # convertir a [1, time] si es mono
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    sr = audio_dict["sampling_rate"]

    # resample a 16kHz si es necesario
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # convertir a mono si tiene varios canales
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    with torch.no_grad():
        embeddings = classifier.encode_batch(waveform)
        xvector = embeddings.squeeze()

    return xvector

# --- Ejemplo de uso ---
audio_data = dataset[0]["audio"]

xvector_tensor = generar_xvector(audio_data)
xvector_np = xvector_tensor.cpu().numpy()
np.save("xvector.npy", xvector_np)

print("✅ X-vector generado correctamente")
print("Shape:", xvector_np.shape)