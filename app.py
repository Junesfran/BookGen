from flask import Flask, send_file
from arranque import funcionamiento_modelo
import os 

app = Flask(__name__)

@app.get('/')
def ver_resultado():
    if not os.path.isfile('./resultados/output.wav'):
        funcionamiento_modelo()
    
    return send_file('./resultados/output.wav')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)