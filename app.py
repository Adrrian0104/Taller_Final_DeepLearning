from flask import Flask, request, jsonify, render_template
import joblib
import re
import json
import os

app = Flask(__name__)

model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Archivo para guardar historial
HISTORIAL_FILE = 'historial.json'

def cargar_historial():
    if os.path.exists(HISTORIAL_FILE):
        with open(HISTORIAL_FILE, 'r') as f:
            return json.load(f)
    return []

def guardar_historial(historial):
    with open(HISTORIAL_FILE, 'w') as f:
        json.dump(historial, f)

def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z0-9\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json()
    texto = datos.get('texto', '')
    texto_limpio = limpiar(texto)
    vector = tfidf.transform([texto_limpio])
    prediccion = model.predict(vector)[0]
    probabilidad = model.predict_proba(vector)[0]
    confianza = round(max(probabilidad) * 100, 2)

    resultado = {
        'prediccion': 'Positivo ✅' if prediccion == 1 else 'Negativo ⛔',
        'confianza': f'{confianza}%',
        'texto': texto[:80] + '...' if len(texto) > 80 else texto
    }

    # Guardar en historial
    historial = cargar_historial()
    historial.append({
        'texto': resultado['texto'],
        'prediccion': resultado['prediccion'],
        'confianza': confianza
    })
    guardar_historial(historial)

    return jsonify(resultado)

@app.route('/estadisticas')
def estadisticas():
    historial = cargar_historial()
    total = len(historial)
    positivos = sum(1 for h in historial if 'Positivo' in h['prediccion'])
    negativos = total - positivos
    confianza_promedio = round(
        sum(h['confianza'] for h in historial) / total, 2
    ) if total > 0 else 0

    return jsonify({
        'total': total,
        'positivos': positivos,
        'negativos': negativos,
        'confianza_promedio': confianza_promedio,
        'historial': historial[-10:]  # últimas 10
    })

if __name__ == '__main__':
    app.run(debug=True)