from pydub import AudioSegment
import os
import sys

# Obtener la ruta de la carpeta donde está este script
base_path = os.path.dirname(os.path.abspath(__file__))

# Configurar las rutas explícitas para ffmpeg y ffprobe
AudioSegment.converter = os.path.join(base_path, "ffmpeg.exe")
AudioSegment.ffprobe   = os.path.join(base_path, "ffprobe.exe")

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
preparar_audio_para_xtts("quijote_vol1_01_cervantes.mp3", "quijote.wav")