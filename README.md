## Integrantes
 - Hugo Quesada
 - Juan Nestor Franco


## Descripción
BookGen es una aplicación de Inteligencia Artificial local que transforma textos en audio personalizado. El problema que intenta resolver es la falta de conexión emocional y la monotonía de las voces sintéticas estándar. Mediante la tecnología de clonación de voz, permite que cualquier usuario suba una muestra de su propia voz (o la de un ser querido) para que la IA narre textos con ese timbre específico, preservando la identidad vocal y la calidez humana.

## Modelo Base
Hemos seleccionado el modelo XTTS v2 de Coqui AI, que se puede encontrar en Hugging Face.

¿Por qué?: Es un modelo multilingüe diseñado específicamente para tareas de Text-to-Speech (TTS) con una altísima fidelidad. Soporta español de forma nativa y destaca por su capacidad "Zero-Shot", lo que significa que puede imitar una voz nueva sin necesidad de re-entrenar el modelo completo.

## Técnica de Adaptación: Ajuste de Contexto (Zero-Shot TTS)
En lugar de un Fine-tuning tradicional, hemos optado por el Ajuste de Contexto mediante vectores de identidad vocal.

¿Por qué esta elección?: El Fine-tuning requiere miles de muestras de audio y horas de procesamiento en GPU. Para un asistente de audiolibros ágil, el ajuste de contexto es más eficiente: el modelo procesa un audio corto (referencia) y extrae sus características (tono, ritmo, timbre) para condicionar la generación del habla en tiempo real.

Cómo funciona: El modelo XTTS v2 utiliza un codificador de voz que convierte 6-10 segundos de audio en un "latente condicional" que guía al decodificador para que el texto resultante suene como la muestra proporcionada.

## Dataset
El "conocimiento" especializado de nuestra app no proviene de una base de datos de texto, sino de un Dataset de Identidad Vocal:

Origen: Audios en datasets de Hugging Face o grabados por los usuarios en formato .m4a o .opus (notas de voz de WhatsApp).

Procesamiento: Los datos se procesan mediante nuestro script src/convertir_audio.py, que utiliza la librería pydub y FFmpeg para normalizar los audios a:

- Formato: WAV (PCM 16-bit).
- Canal: Mono (único).
- Frecuencia: 22050 Hz (estándar nativo del modelo XTTSv2).
