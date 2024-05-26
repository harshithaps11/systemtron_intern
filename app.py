import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(file_path):
    # Load the diabetes dataset
    diabetes_dataset = pd.read_csv(file_path)
    return diabetes_dataset

def preprocess_data(dataset):
    # separating the data and labels
    X = dataset.drop(columns='Outcome', axis=1)
    Y = dataset['Outcome']
    
    # Standardize the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    
    return X, Y, scaler

def train_model(X_train, Y_train, kernel='linear'):
    # Train the SVM classifier
    classifier = SVC(kernel=kernel)
    classifier.fit(X_train, Y_train)
    return classifier

def evaluate_model(classifier, X_train, Y_train, X_test, Y_test):
    # Accuracy score on the training data
    X_train_prediction = classifier.predict(X_train)
    training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
    print('Accuracy score of the training data : ', training_data_accuracy)

    # Accuracy score on the test data
    X_test_prediction = classifier.predict(X_test)
    test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
    print('Accuracy score of the test data : ', test_data_accuracy)

def predict_diabetes(input_data, scaler, classifier):
    # Standardize the input data
    input_data_reshaped = np.asarray(input_data).reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = classifier.predict(std_data)

    # Print prediction result
    if prediction[0] == 0:
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

def get_user_input():
    # Get input from user
    pregnancies = float(input("Enter number of pregnancies: "))
    glucose = float(input("Enter plasma glucose concentration (mg/dL): "))
    blood_pressure = float(input("Enter diastolic blood pressure (mm Hg): "))
    skin_thickness = float(input("Enter triceps skin fold thickness (mm): "))
    insulin = float(input("Enter insulin level (mu U/ml): "))
    bmi = float(input("Enter BMI (body mass index): "))
    diabetes_pedigree = float(input("Enter diabetes pedigree function: "))
    age = float(input("Enter age (years): "))

    return pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age

def main():
    # Load data
    file_path = 'diabetes.csv'
    diabetes_dataset = load_data(file_path)

    # Preprocess data
    X, Y, scaler = preprocess_data(diabetes_dataset)

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Train model
    classifier = train_model(X_train, Y_train)

    # Evaluate model
    evaluate_model(classifier, X_train, Y_train, X_test, Y_test)

    # Get user input
    input_data = get_user_input()

    # Make prediction
    predict_diabetes(input_data, scaler, classifier)
    


if __name__ == "__main__":
    main()
