#basic python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Prophet Library
from prophet import Prophet

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