import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import pickle

# Load the trained model
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

# Function to predict car price
def predict_car_price():
    # Prompt user for input
    car_name = input("Enter the car's name: ").capitalize()
    company = input("Enter the company: ").capitalize()
    year = int(input("Enter the year: "))
    kms_driven = int(input("Enter the kilometers driven: "))
    fuel_type = input("Enter the fuel type: ").capitalize()

    # Prepare input data for prediction
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                               data=np.array([car_name, company, year, kms_driven, fuel_type]).reshape(1, -1))

    # Check categories used during training
    print("Categories used during training:")
    categories = pipe.steps[0][1].transformers_[0][1].categories
    print(categories)

    # Check categories in input data
    print("Categories in input data:")
    for i, col in enumerate(input_data.columns):
        print(f"{col}: {input_data.iloc[0, i]}")

    # Predict car price
    predicted_price = pipe.predict(input_data)
    print("Predicted Price:", predicted_price[0])

# Predict car price
predict_car_price()
