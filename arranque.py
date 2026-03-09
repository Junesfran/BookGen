import os
# from descarga_contenido import descargar_modelos, descargar_chileno
from generador_wav import generar_wav

def funcionamiento_modelo(entrenamiento: str):
    # if not os.path.exists('./models'):
    #     descargar_modelos()
    
    generar_wav(entrenamiento)    

    
        