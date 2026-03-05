from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset

# Descargar
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

datasetFem = load_dataset("ylacombe/google-chilean-spanish", 'female')
datasetMal = load_dataset("ylacombe/google-chilean-spanish", 'male')

# Guardar en carpetas locales
processor.save_pretrained("./models/speecht5_tts")
model.save_pretrained("./models/speecht5_tts")
vocoder.save_pretrained("./models/speecht5_hifigan")

datasetFem.save_to_disk("./datasets/chileno/female")
datasetMal.save_to_disk("./datasets/chileno/male")