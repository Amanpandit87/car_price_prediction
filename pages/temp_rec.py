import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load cleaned car dataset
df1 = pd.read_csv("Cleaned_Car_Data.csv")

# Create the pivot table
pivot_car = df1.pivot_table(columns=["Fuel Type", "Year", "Suspension", "kms Driven"], 
                            index="Price", values="Car Model", aggfunc=lambda x: len(x.unique()))
pivot_car.fillna(0, inplace=True)

# Convert pivot table to sparse matrix
sparse_car = csr_matrix(pivot_car)

# Create and fit the recommendation model
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_car)

# Function to recommend car models
def recommend_car_models(car_price):
    car_id = np.where(pivot_car.index == car_price)[0][0]
    distance, suggestion = model.kneighbors(pivot_car.iloc[car_id, :].values.reshape(1, -1), n_neighbors=6)
    
    recommended_car_models = []
    for i in range(1, len(suggestion[0])):  # Start from 1 to avoid showing the same car
        suggested_price = pivot_car.index[suggestion[0][i]]
        
        # Check if the car price is in df1["Price"]
        if suggested_price in df1["Price"].values:
            # Get the corresponding car model
            car_model = df1[df1["Price"] == suggested_price]["Car Model"].values[0]
            recommended_car_models.append(car_model)
    
    return recommended_car_models

# Streamlit UI
st.set_page_config(page_title="Car Model Recommendation System", page_icon="🚗", layout="centered")

# Title
st.markdown("<h1 style='text-align:center; color:#219C90;'>Car Model Recommendation System</h1>", unsafe_allow_html=True)

# User Input: Car Price
car_price = st.number_input("Enter the car price to find similar car models:", min_value=int(pivot_car.index.min()), max_value=int(pivot_car.index.max()), step=1000)

# Button to get recommendations
if st.button("Get Recommendations"):
    if car_price not in pivot_car.index:
        st.error(f"Error: Car price '{car_price}' not found in dataset!")
    else:
        recommended_car_models = recommend_car_models(car_price)
        if recommended_car_models:
            st.write("### Recommended Car Models:")
            for model in recommended_car_models:
                car_recommended = st.write(f"- {model}")
            st.markdown('<a href="temp" target="_self">car_recommended</a>', unsafe_allow_html=True)
        else:
            st.write("No recommendations found.")

# Save the recommendation model
with open("car_new_price_recommendation.pkl", "wb") as f:
    pickle.dump((model, pivot_car), f)

#st.success("Model saved as 'car_new_price_recommendation.pkl'")

#st.title("Second Page")

#st.write("This is the second page.")

# Button to navigate back to main page
#if st.button("Back to Main Page"):
#    st.markdown('<a href="temp" target="_self">st.button("Back to Main Page")</a>', unsafe_allow_html=True)

