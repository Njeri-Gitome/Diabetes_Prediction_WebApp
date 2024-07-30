import numpy as np
import pickle
import streamlit as st

# Load the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make prediction
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Return result based on prediction
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
  

    # Title for the user page
    st.title('Diabetes Prediction WebApp')

    # Input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the person')
    
    # Ensure all fields are filled
    if st.button('Diabetes Test Result'):
        if Pregnancies and Glucose and BloodPressure and SkinThickness and Insulin and BMI and DiabetesPedigreeFunction and Age:
            try:
                # Convert inputs to float
                input_data = [
                    float(Pregnancies),
                    float(Glucose),
                    float(BloodPressure),
                    float(SkinThickness),
                    float(Insulin),
                    float(BMI),
                    float(DiabetesPedigreeFunction),
                    float(Age)
                ]
                
                # Make prediction
                diagnosis = diabetes_prediction(input_data)
                
                # Display result
                st.success(diagnosis)
            except ValueError:
                st.error('Please enter valid numerical values.')
        else:
            st.error('Please fill in all the fields.')
    
 

if __name__ == '__main__':
    main()
