
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# 1. Cargar el modelo y los encoders (traductores)
model = joblib.load('../models/model.pkl')
le_item = joblib.load('../models/le_item.pkl')
le_cat = joblib.load('../models/le_cat.pkl')
le_target = joblib.load('../models/le_target.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    
    if request.method == 'POST':
        try:
            # Obtener datos del formulario web
            item_input = request.form['item']
            cat_input = request.form['category']

            # Transformar el texto ingresado a los números que el modelo conoce
            # Usamos transform() con los encoders que guardamos en el Paso 2
            item_encoded = le_item.transform([item_input])[0]
            cat_encoded = le_cat.transform([cat_input])[0]

            # Realizar la predicción (devuelve un ID numérico)
            pred_id = model.predict([[item_encoded, cat_encoded]])[0]
            
            # Convertir ese ID de vuelta al nombre del Proveedor Real
            prediction = le_target.inverse_transform([pred_id])[0]
            
        except ValueError:
            error = "El artículo o categoría no se encuentran en el historial. Intenta con 'Laptop' y 'Electronics'."
        except Exception as e:
            error = f"Ocurrió un error: {str(e)}"

    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)