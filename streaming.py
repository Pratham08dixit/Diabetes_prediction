import numpy as np
import pickle
import streamlit as st


#LOADING THE SAVED MODEL
loaded_model=pickle.load(open('C:/Users\prath\OneDrive\Desktop\machine_learning/trained_model.sav','rb'))


def diabetes_prediction(input_data):

    # CHANGING THE INPUT DATA TO NUMPY ARRAY
    ipdata_as_nparray = np.asarray(input_data)

    # RESHAPE THE ARRAY AS WE ARE PREDICTING FOR ONE INSTANCE
    ipdata_reshape = ipdata_as_nparray.reshape(1, -1)

    pred = loaded_model.predict(ipdata_reshape)
    print(pred)

    if (pred[0] == 0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"


#def main():
def main( ):
    st.title('Diabetes prediction web app')

    #getting imput from the user

    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('BP Value')
    SkinThickness = st.text_input('thicknoess of skin')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI of the person')
    DiabetesPedigreeFunction = st.text_input('DBF value')
    Age = st.text_input('Age of the person')



    #code for prediction
    diagnosis = ''

    #creatin a button for prediction
    if st.button('diabetes_test_result'):
        diagnosis = diabetes_prediction([Pregnancies ,Glucose , BloodPressure , SkinThickness ,  Insulin ,  BMI , DiabetesPedigreeFunction , Age])

    st.success(diagnosis)



if __name__ == '__main__':
    main()




