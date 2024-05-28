from flask import Flask, request, render_template
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time



app = Flask(__name__)

# Load model
model = joblib.load('extra_tree_model_mobil123.pkl')
label_encoders = joblib.load('label_encoders.pkl')
if 'daerah' in label_encoders:
    del label_encoders['daerah']
scaler = joblib.load('scaler.pkl')



@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/data_mobil_bekas')
# def data_mobil_bekas():
#     cars = scrape_mobil_bekas()
#     for car in cars:
#         print(car)  # Debugging print to check the data
#     return render_template('data_mobil_bekas.html', cars=cars)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Dapatkan data dari form
        data = request.form.to_dict()
        
        # Convert data to DataFrame
        df_input = pd.DataFrame([data])
        
        # Melakukan Label Encoding pada kolom-kolom kategorikal
        for col in ['warna', 'bahan_bakar', 'dirakit', 'transmisi', 'jenis_mobil', 'brand']:
            if col in df_input.columns:
                # Gunakan encoder yang sama seperti saat training
                le = label_encoders[col]
                df_input[col] = le.transform(df_input[col])
        
        # Normalisasi data
        df_input_normalized = scaler.transform(df_input)
        
        # Prediksi
        prediction = model.predict(df_input_normalized)
        
        return render_template('predict.html', prediction=prediction[0])
    return render_template('predict.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/car_list')
def car_list():
    # Placeholder: Replace with actual logic to fetch car data
    car_data = [
        {"brand": "Toyota", "model": "Corolla", "predicted_price": 200000000},
        {"brand": "Honda", "model": "Civic", "predicted_price": 250000000},
        # Add more car data here
    ]
    return render_template('car_list.html', car_data=car_data)

if __name__ == '__main__':
    app.run(debug=True)

# # Fungsi untuk scraping data mobil bekas
# def scrape_mobil_bekas():
#     options = webdriver.ChromeOptions()
#     options.add_argument('--headless')
#     options.add_argument('--disable-gpu')
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')

#     driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)
#     driver.get('https://www.mobil123.com/mobil-bekas-dijual/indonesia')
#     time.sleep(5)  # Tunggu hingga halaman selesai dimuat

#     cars = []
#     card_count = 0
#     while card_count < 1000:
#         cards = driver.find_elements(By.CSS_SELECTOR, 'article')
#         if not cards:
#             print("No cards found. Exiting loop.")
#             break
#         for card in cards:
#             if card_count >= 1000:
#                 break
#             try:
#                 car = {}
#                 car['link'] = card.find_element(By.CSS_SELECTOR, 'a.ellipsize').get_attribute('href')
#                 car['nama_mobil'] = card.find_element(By.CSS_SELECTOR, 'h2').text.strip()
#                 car['harga'] = card.find_element(By.CSS_SELECTOR, 'h3.u-text-3').text.strip()
#                 car['tahun_kendaraan'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(2) span.u-text-bold.u-block').text.strip()
#                 car['kilometer'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(3) span.u-text-bold.u-block').text.strip()
#                 car['warna'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(4) span.u-text-bold.u-block').text.strip()
#                 car['transmisi'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(6) span.u-text-bold.u-block').text.strip()
#                 car['mesin_enginecc'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(5) span.u-text-bold.u-block').text.strip()
#                 car['bahan_bakar'] = card.find_element(By.CSS_SELECTOR, 'div:nth-of-type(10) span.u-text-bold').text.strip()
#                 car['daerah'] = card.find_element(By.CSS_SELECTOR, 'span.c-chip--wrap:nth-of-type(2)').text.strip()
#                 cars.append(car)
#                 card_count += 1
#             except Exception as e:
#                 print(f"Error processing card: {e}")
#                 continue

#         try:
#             # Check for and click the "Next" button to load more results
#             next_button = driver.find_element(By.CSS_SELECTOR, '.pagination a[rel="next"]')
#             next_button.click()
#             time.sleep(5)  # Tunggu hingga halaman berikutnya selesai dimuat
#         except Exception as e:
#             print(f"No more pages or error: {e}")
#             break

#     driver.quit()
#     return cars