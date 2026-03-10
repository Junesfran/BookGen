from flask import Flask, send_file, request, redirect
from arranque import funcionamiento_modelo
from descarga_contenido import bajar_modelo
import os 

app = Flask(__name__)

@app.get('/favicon.ico')
def nada():
    return 'Aqui no hay nada que ver', 418

@app.get('/<entrenamiento>')
def ver_resultado(entrenamiento: str):
    
    if not os.path.isfile('./output/resultado.wav'):
        funcionamiento_modelo(entrenamiento)
    
    return send_file('./output/resultado.wav')

@app.get('/recargar/<entrenamiento>')
def volver_lanzar(entrenamiento: str):
    funcionamiento_modelo(entrenamiento)
    
    return redirect('/<entrenamiento>/' + entrenamiento)
    

@app.post("/subir/<archivo>")
def recibir_referencia(archivo: str):
    archivo_premium = request.files["archivo"]
    
    ruta_guardado = os.path.join('.','datasets','referencias', archivo)
    
    with open(ruta_guardado, "wb") as f:
        f.write(archivo_premium.read())
        
    return 'Archivo bien recibido', 418

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)