#app code

import numpy as np
import pickle
import streamlit as st

#load the model
with open("survival_model.sav", "rb") as file:
    loaded_model = pickle.load(file)

def survival(input_data):
    inp_data_array = np.array(input_data)

    inp_data_reshaped = inp_data_array.reshape(1, -1)

    pred_in = loaded_model.predict(inp_data_reshaped)
    # print(pred_in)

    if pred_in[0] == 0:
        return "The Person did not SURVIVE"
    else:
        return "The Person Survived"


def main():
    # set title
    st.title("Titanic Survival Web Application")

    # set header
    st.header("Enter the details of a passenger")
    
    # input values from the user
    Pclass = st.text_input("Enter Pclass (1, 2, 3)")
    Sex = st.text_input("Enter Sex of a preson(1:male or 2:female)")
    Age = st.text_input("Enter Age of the person(0-80)")
    SibSp = st.text_input("Enter SibSp (0-5)")
    Parch = st.text_input("Enter Parch (0-6)")
    Fare = st.text_input("Enter Fare (0-512)")
    Embarked = st.text_input("Enter Embarked from which place (0, 1, 2)")

    # code for prediction
    answer = ""                  # empty string here the answer will be stored

    #Creating a button for prediction
    if st.button("Predict Survival"):
        answer = survival([Pclass, Sex, Age, SibSp, Parch, Fare, Embarked])

    st.success(answer)

if __name__ == '__main__':
    main() 
