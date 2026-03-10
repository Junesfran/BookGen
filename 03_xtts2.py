import torch
from TTS.api import TTS
import os
from datasets import load_dataset
import soundfile as sf # Necesario para guardar el audio de referencia

# 1. Definir el dispositivo (Cuda si tienes Nvidia, si no CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# 2.1 Cargar el modelo (Esto descargará ~2GB la primera vez)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 2.2 En caso de tener el modelo en local
#path_al_modelo = 'C:/Users/Vespertino/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2'
#tts = TTS(model_path=path_al_modelo, config_path=os.path.join(path_al_modelo, "config.json")).to(device)

os.makedirs("./output", exist_ok=True)

# 3. Cargar el dataset chileno
print("Cargando dataset...")
dataset_chileno = load_dataset("ylacombe/google-chilean-spanish", "female", split="train", streaming=False)

# 4. Extraer el primer audio y guardarlo como temporal para XTTS
# XTTS necesita un archivo físico, no un objeto de memoria
ejemplo_audio = dataset_chileno[0]["audio"]
ruta_temporal_ref = "referencia_chilena.wav"
sf.write(ruta_temporal_ref, ejemplo_audio["array"], ejemplo_audio["sampling_rate"])

texto_a_decir = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún palomino de añadidura los domingos, consumían las tres partes de su hacienda. El resto della concluían sayo de velarte, calzas de velludo para las fiestas, con sus pantuflos de lo mesmo, y los días de entresemana se honraba con su vellorí de lo más fino. Tenía en su casa una ama que pasaba de los cuarenta y una sobrina que no llegaba a los veinte, y un mozo de campo y plaza que así ensillaba el rocín como tomaba la podadera. Frisaba la edad de nuestro hidalgo con los cincuenta años. Era de complexión recia, seco de carnes, enjuto de rostro, gran madrugador y amigo de la caza. Quieren decir que tenía el sobrenombre de «Quijada», o «Quesada», que en esto hay alguna diferencia en los autores que deste caso escriben, aunque por conjeturas verisímilesII se deja entender que se llamaba «Quijana». Pero esto importa poco a nuestro cuento: basta que en la narración dél no se salga un punto de la verdad."

# 5. Generar el audio
print("Generando voz... ten paciencia, XTTS es pesado.")
tts.tts_to_file(
    text=texto_a_decir,
    #speaker_wav=ruta_temporal_ref, # Ahora sí es una ruta válida
    speaker_wav='quijote.wav',
    language="es",
    file_path="./output/resultado.wav"
)

print("¡Éxito! Audio generado en ./output/resultado.wav")