from flask import Flask, request, render_template
import pandas as pd
from pycaret.regression import load_model, predict_model

app = Flask(__name__)

# Load model
model = load_model('models/my_best_pipeline')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            'brand': request.form['brand'],
            'jenis_mobil': request.form['jenis_mobil'],
            'tahun_kendaraan': int(request.form['tahun_kendaraan']),
            'warna': request.form['warna'],
            'transmisi': request.form['transmisi'],
            'kilometer': int(request.form['kilometer']),
            'mesin_enginecc': int(request.form['mesin_enginecc']),
            'bahan_bakar': request.form['bahan_bakar'],
            'dirakit': request.form['dirakit'],
            'penumpang': int(request.form['penumpang']),
            'pintu': int(request.form['pintu'])
        }
        
        df = pd.DataFrame([data])
        
        # Predict car price
        prediction = predict_model(model, data=df)
        predicted_price = prediction['prediction_label'][0]
        
        return render_template('predict.html', prediction_text='Predicted Price: Rp {:,.2f}'.format(predicted_price))
    return render_template('predict.html')

@app.route('/car_list')
def car_list():
    # Load data from a CSV file or database
    data = pd.read_csv('wow.csv')
    cars = data.to_dict(orient='records')
    return render_template('car_list.html', cars=cars)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
