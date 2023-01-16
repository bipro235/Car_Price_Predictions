from flask import Flask, render_template, request
import pandas as pd
import pickle
app = Flask(__name__)
import numpy as np

model = pickle.load(open("Price_predictor_model_2.pkl", 'rb'))
car = pd.read_csv("Cleaned Car Data.csv")

@app.route('/')
def index():
    print("Hellow")
    Car_Name = sorted(car['car_name'].unique())
    Car_Model = sorted(car['model'].unique())
    Car_Brand = sorted(car['brand'].unique())
    Min_Cost_Price = sorted(car['min_cost_price'].unique())
    Max_Cost_Price = sorted(car['max_cost_price'].unique())
    Vehicle_Age = sorted(car['vehicle_age'].unique())
    Km_Driven = sorted(car['km_driven'].unique())
    Seller_Type = sorted(car['seller_type'].unique())
    Fuel_Type = sorted(car['fuel_type'].unique())
    Transmission_Type = sorted(car['transmission_type'].unique())
    Car_Mileage = sorted(car['mileage'].unique())
    Engine_CC = sorted(car['engine'].unique())
    Max_Power = sorted(car['max_power'].unique())
    Numner_of_seats = sorted(car['seats'].unique())

    return render_template('index.html', car_name=Car_Name, brand=Car_Brand, model = Car_Model, min_cost_price=Min_Cost_Price,
                           max_cost_price=Max_Cost_Price, vehicle_age=Vehicle_Age,
                           km_driven=Km_Driven, seller_type=Seller_Type,
                           fuel_type=Fuel_Type, transmission_type=Transmission_Type,
                           mileage=Car_Mileage, engine=Engine_CC, max_power=Max_Power,
                           seats=Numner_of_seats)


@app.route('/predict', methods= ['POST'])
def predict():

    Brand = request.form.get('brand')
    Car_name = request.form.get('car_name')
    Model = request.form.get('model')
    Min_cost_price = int(request.form.get('min_cost_price'))
    Max_cost_price = int(request.form.get('max_cost_price'))
    Vehicle_age = int(request.form.get('vehicle_age'))
    Km_driven = int(request.form.get('km_driven'))
    Seller_type = request.form.get('seller_type')
    Fuel_type = request.form.get('fuel_type')
    Transmission_type = request.form.get('transmission_type')
    Mileage = int(request.form.get('mileage'))
    Engine = int(request.form.get('engine'))
    Max_power = int(request.form.get('max_power'))
    Seats = int(request.form.get('seats'))

    print(Brand, Car_name, Min_cost_price, Max_cost_price, Seller_type, Vehicle_age, Km_driven, Fuel_type)

    prediction = model.predict(pd.DataFrame([[Car_name, Brand, Model, Min_cost_price, Max_cost_price, Vehicle_age, Km_driven, Seller_type, Fuel_type, Transmission_type, Mileage, Engine, Max_power, Seats]], columns=['car_name', 'brand', 'model', 'min_cost_price', 'max_cost_price','vehicle_age', 'km_driven', 'seller_type', 'fuel_type','transmission_type', 'mileage', 'engine', 'max_power', 'seats']))

    return str(prediction[0])



if __name__ == "__main__":
    app.run(debug=True)
