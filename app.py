import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

# Load dataset
diabetes = pd.read_csv('diabetes.csv')

# Streamlit App Title
st.title('Diabetes Prediction App')

# Sidebar Navigation
st.sidebar.header('Navigation')
st.sidebar.write('Select the following options to navigate through the app')
st.sidebar.write('---')
options = st.sidebar.radio('Choose a Section:', ['Overview', 'Dataset Visualization', 'Predict Diabetes', 'Model Evaluation'])

if options == 'Overview':
    st.subheader('Project Overview')
    st.write('''
             This is a diabetes prediction app that predicts whether a person has diabetes or not based on their health parameters.

             **Dataset** : PIMA Diabetes Database

             **Model** : Support Vector Machine (SVM)

             **Features** : Blood Pressure, Glucose Levels, BMI, Age, and more...
             ''')

elif options == 'Dataset Visualization':
    st.subheader('Exploratory Data Analysis')
    st.write('**Dataset Information**')
    st.write(diabetes.head())
    st.write('### Dataset Shape')
    st.write(diabetes.shape)
    st.write('### Statistical Summary')
    st.write(diabetes.describe())
    st.write('### Data Outcome Overview')
    st.write(diabetes['Outcome'].value_counts())
    st.write('### Outcome Distribution')
    fg, ax = plt.subplots()
    sns.countplot(x='Outcome', data=diabetes, palette='viridis', ax=ax)
    st.pyplot(fg)
    st.write('### Correlation Heatmap')
    corr = diabetes.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

elif options == 'Predict Diabetes':
    st.subheader('Diabetes Prediction')

    # Input format
    st.write('### Enter your health parameters:')
    pregnancies = st.number_input('Pregnancies', min_value=0, step=1)
    glucose = st.number_input('Glucose', min_value=0, step=1)
    blood_pressure = st.number_input('Blood Pressure', min_value=0, step=1)
    skin_thickness = st.number_input('Skin Thickness', min_value=0)
    insulin = st.number_input('Insulin', min_value=0, step=1)
    bmi = st.number_input('BMI', min_value=0.0, value=1.0, format="%.2f")
    age = st.number_input('Age', min_value=0, step=1)
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, format='%.3f')

    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age)
    input_data_as_numpy = np.asarray(input_data)

    # Reshape the input data
    input_data_reshaped = input_data_as_numpy.reshape(1, -1)

    # Preprocess the data and train the model once
    X = diabetes.drop('Outcome', axis=1)
    Y = diabetes['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear', probability=True)  # Added probability for robustness
    classifier.fit(X_train, Y_train)

    # Standardize the user input data
    standard_data = scaler.transform(input_data_reshaped)

    if st.button('Predict'):
        prediction = classifier.predict(standard_data)
        st.write('### Prediction:')
        if prediction[0] == 0:
            st.write('You have No diabetes')
        else:
            st.write('You have diabetes')

elif options == 'Model Evaluation':
    st.subheader('Model Evaluation Metrics')
    X = diabetes.drop('Outcome', axis=1)
    Y = diabetes['Outcome']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, stratify=Y, random_state=2)
    classifier = svm.SVC(kernel='linear')
    classifier.fit(X_train, Y_train)
    Y_Pred_train = classifier.predict(X_train)
    Y_Pred_test = classifier.predict(X_test)
    train_accuracy = accuracy_score(Y_train, Y_Pred_train)
    test_accuracy = accuracy_score(Y_test, Y_Pred_test)
    st.write(f'Training Accuracy: {train_accuracy:.2f}')
    st.write(f'Test Accuracy: {test_accuracy:.2f}')
    st.write('### Classification Report')
    r = classification_report(Y_test, Y_Pred_test, output_dict=True)
    r_df = pd.DataFrame.from_dict(r).transpose()
    st.dataframe(r_df)
    st.write('### Confusion Matrix')
    cm = confusion_matrix(Y_test, Y_Pred_test)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)
