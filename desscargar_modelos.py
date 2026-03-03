from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

# Descargar
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Guardar en carpetas locales
processor.save_pretrained("./models/speecht5_tts")
model.save_pretrained("./models/speecht5_tts")
vocoder.save_pretrained("./models/speecht5_hifigan")