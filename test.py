import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split data into training and testing sets
X = data.drop(columns="Class", axis=1)
y = data["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model performance
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

# Web app
st.title("Credit Card Fraud Detection Model")

st.write(f"Training Accuracy: {train_acc:.2f}")
st.write(f"Testing Accuracy: {test_acc:.2f}")

input_df = st.text_input('Enter all required feature values separated by commas')

submit = st.button("Submit")

if submit:
    try:
        # Split the input string into a list
        input_df_splited = input_df.split(',')
        
        # Convert list to numpy array
        np_df = np.asarray(input_df_splited, dtype=np.float64)
        
        # Ensure the input length matches the number of features
        if np_df.shape[0] == X.shape[1]:
            # Reshape and predict
            prediction = model.predict(np_df.reshape(1, -1))
            
            # Display the result
            if prediction[0] == 0:
                st.write("Legitimate Transaction")
            else:
                st.write("Fraudulent Transaction")
        else:
            st.write(f"Please enter exactly {X.shape[1]} feature values.")
    except ValueError:
        st.write("Invalid input. Please enter numeric values separated by commas.")
