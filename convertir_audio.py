from pydub import AudioSegment
import os

AudioSegment.converter = "ffmpeg.exe" 

def preparar_audio_para_xtts(ruta_entrada, ruta_salida):
    # 1. Cargar el archivo (detecta m4a, mp3, etc.)
    audio = AudioSegment.from_file(ruta_entrada)

    # 2. Ajustar a los estándares óptimos de XTTS:
    # - Mono (1 canal)
    # - 22050 Hz (frecuencia de muestreo estándar de Coqui)
    audio = audio.set_channels(1).set_frame_rate(22050)

    # 3. Exportar como WAV
    audio.export(ruta_salida, format="wav")
    print(f"✅ Audio convertido con éxito: {ruta_salida}")

# Uso:
preparar_audio_para_xtts("mi_voz_movil.m4a", "clonacion_voz.wav")