import sys
import torch
import numpy as np

from datasets import load_dataset

try:
    print("Loading speaker embeddings...")
    
    embeddings_dataset = load_dataset(
        'parquet',
        data_files="./datasets/euskera.parquet",
        split='train'
    )
    # Pick one speaker embedding (index 7306 is a commonly used female voice)
    speaker_embedding = torch.tensor(
        embeddings_dataset[0]["speaker_embeddings"]
    ).unsqueeze(0)
    
    print(f"Speaker embedding shape: {speaker_embedding.shape}")
except Exception as e:
    print(f"Error loading speaker embeddings: {e}")
    sys.exit(1)


recorer_json = {
  "escenas": [
    {
      "personaje": "Bernarda",
      "texto": "¡Silencio! ¿Es que no se puede estar en esta casa con un poco de paz?",
      "emocion": "ira"
    },
    {
      "personaje": "Adela",
      "texto": "¡Yo no aguantaré más tus gritos! Me iré de aquí, aunque sea al arroyo.",
      "emocion": "desafiante"
    },
    {
      "personaje": "La Poncia",
      "texto": "Bernarda, deja a la muchacha. La sangre se le sube a la cabeza con este calor que hace.",
      "emocion": "calma"
    }
  ]
}

escena = recorer_json["escenas"]
voces = len(embeddings_dataset)

for x in escena:
    rn = np.random.randint(0, voces)
    x['voz'] = rn
    
print(escena)