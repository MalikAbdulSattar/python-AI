import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Get the DataFrame
dataset=pd.read_csv("Copy of sonar data.csv",header=None)
dataset.head()

#prepare the dataset
dataset.groupby(60).mean()
dataset.describe()

#Train test split  
x=dataset.drop(columns=60,axis=1)
y=dataset[60]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

#Train and Evaluate Model

model= LogisticRegression()
model.fit(x_train, y_train)
train_pred=model.predict(x_train)
print(accuracy_score(train_pred,y_train))
test_pred=model.predict(x_test)
print(accuracy_score(test_pred,y_test))

#Web Site 

st.title("Sonar Rock Vs Mine Prediction")
input_data=st.text_input("Enter Here!")
if st.button("Predict"):
    input_data_arr=np.asarray(input_data.split(","),dtype=float)
    input_data_reshape=input_data_arr.reshape(1,-1)
    prediction=model.predict(input_data_reshape)
    if prediction[0]=='R':
        st.write("The entered data belongs to Rock")
    else:
        st.write("The entered data belongs to Mine")
