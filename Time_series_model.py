#basic python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Prophet Library
from prophet import Prophet

#streamlit library
import streamlit as st

# Functions
## Split data per store
def store_split_data(dataset, store):
    store_data = dataset[dataset["store"] == store]
    return store_data

def get_store_data(dataset):
    n_product = {}
    for i in range(10): 
        product_split = dataset[dataset["product"] == i] 
        n_product[f"n_sold_product_{i}"] = product_split
    
    return n_product  # Return dictionary of DataFrames

def cleaned_store_data(df, store_number, product_number):
    store = store_split_data(df, store_number) #split the main data to stores
    store = get_store_data(store) #get the store data where it will be splitted into product categories
    store = store[f"n_sold_product_{product_number}"] #get the desired product number
    store = store.drop(columns = ["store", "product"]) #this data will be passed to the prophet
    store.rename(columns={"Date":"ds", "number_sold":"y"}, inplace = True) #rename the columns so that prophet will accept it
    return store

#Making Forecast function
def make_forecast(df, store, product, forecast):
    #initiate the model
    model = Prophet(changepoint_prior_scale = 0.5)

    #fit the model
    store_data = cleaned_store_data(df, store, product)
    model.fit(store_data)

    #predict future
    future = model.make_future_dataframe(periods=forecast) #1 year prediction
    forecast = model.predict(future)

    #plot the forecast
    fig = model.plot(forecast)
    return fig

#main function
def main():
    st.title("Time Series Forecasting Using Prophet")
    df = st.file_uploader('Upload your csv file', type="csv", accept_multiple_files=False)
    df = pd.read_csv(df, sep = ",")
    store = st.slider('store', min_value=0, max_value=6, value=1)
    product = st.slider('product', min_value=0, max_value=9, value=1)
    forecast_days = st.slider("forecast_days", min_value=0, max_value=365, value=1)

    if st.button('Make Prediction'):
        fig = make_forecast(df, store, product, forecast_days)
        fig.title(f"Stock Prediction for {store} product {product}")
        st.pyplot(fig)
        
if __name__ == "__main__":
    main()
