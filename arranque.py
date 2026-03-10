import os
from descarga_contenido import bajar_modelo
from generador_wav import generar_wav

def funcionamiento_modelo(entrenamiento: str):
    if not os.path.exists('./models/xtts_v2'):
        bajar_modelo()
    
    generar_wav(entrenamiento)    

    
        