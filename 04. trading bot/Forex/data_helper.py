
# Import the libraries
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT 
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import numpy as np
import pandas as pd
import time


# Function for adding more features to the dataset
def add_features(df, window_size = 500, get_last_n_rows = None):
    
    # Loop over "Sell" and "Buy" string
    for i_type in ["Sell", "Buy"]:
        
        # Spread
        df["Spread"] = pd.DataFrame(df["Buy"] - df["Sell"])
        
        # Price change
        df["Change - {}".format(i_type)] = df[i_type].pct_change()
        
        # Simple Moving Average (SMA)
        df["Simple Moving Average (SMA) - {}".format(i_type)] = df[i_type].rolling(window = window_size).mean()
        
        # Exponential Moving Average (EMA)
        df["Exponential Moving Average (EMA) - {}".format(i_type)] = df[i_type].ewm(com = window_size).mean()
        
        # Bollinger Bands (Upper)
        df["Bollinger Bands (Upper) - {}".format(i_type)] = df["Simple Moving Average (SMA) - {}".format(i_type)] + (df[i_type].rolling(window = window_size).std() * 2)
        
        # Bollinger Bands (Lower)
        df["Bollinger Bands (Lower) - {}".format(i_type)] = df["Simple Moving Average (SMA) - {}".format(i_type)] - (df[i_type].rolling(window = window_size).std() * 2)
        
        # Standard Deviation (STD)
        df["Standard Deviation (STD) - {}".format(i_type)] = df[i_type].rolling(window = window_size).std()
        
        # Sharpe Ratio
        df["Sharpe Ratio - {}".format(i_type)] = df["Change - {}".format(i_type)].rolling(window = window_size).mean() / df["Change - {}".format(i_type)].rolling(window = window_size).std()
        
        ### Relative Strength Index (RSI)

        # Get the difference in price from previous step
        delta = df[i_type].diff()

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        # Calculate the EWMA
        roll_up1 = up.ewm(span = window_size).mean()
        roll_down1 = down.abs().ewm(span = window_size).mean()

        # Calculate the RSI based on EWMA
        RS1 = roll_up1 / roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))

        # Add to the pandas
        df["Relative Strength Index (RSI) via EWMA - {}".format(i_type)] = RSI1

        # Calculate the SMA
        roll_up2 = up.rolling(window_size).mean()
        roll_down2 = down.abs().rolling(window_size).mean()

        # Calculate the RSI based on SMA
        RS2 = roll_up2 / roll_down2
        RSI2 = 100.0 - (100.0 / (1.0 + RS2))

        # Add to pandas
        df["Relative Strength Index (RSI) via SMA - {}".format(i_type)] = RSI2

        ### Fibonacci Retracement

        # Find the maximum price per window size
        price_max = df[i_type].rolling(window = window_size).max()

        # Find the minimum price per window size
        price_min = df[i_type].rolling(window = window_size).min()

        # Fibonacci Levels considering original trend as upward move
        diff = price_max - price_min
        level1 = price_max - 0.236 * diff
        level2 = price_max - 0.382 * diff
        level3 = price_max - 0.618 * diff

        # Add to dataframe
        df["Fibonacci Retracement (Price Min) - {}".format(i_type)] = price_min
        df["Fibonacci Retracement (Level 1) - {}".format(i_type)] = level1
        df["Fibonacci Retracement (Level 2) - {}".format(i_type)] = level2
        df["Fibonacci Retracement (Level 3) - {}".format(i_type)] = level3
        df["Fibonacci Retracement (Price Max) - {}".format(i_type)] = price_max
        
    # Remove the first window_size rows
    df = df.iloc[window_size:, :].reset_index().iloc[:, 1:]
    
    # If get_last_n_rows is defined
    if (type(get_last_n_rows) == int):
        
        # Get only last N rows
        df = df.iloc[get_last_n_rows:, :]
        
    # Remove the "ticker" column
    df = df.drop(labels = ["Ticker"], axis = 1)

    return df


# Gather one row of dataset in real time
def gather_data_online(driver):

    # Get the row data
    rowsData = driver.find_elements(by = "class name", value = "table__row-body")

    # Get the dataset for current timestep
    dataset_current = pd.DataFrame([rowsData[index].text.split() for index in range(8)],
                               columns = ["ticker", "sell", "buy", "change", "change_percentage"])[["ticker", "sell", "buy"]]
    
    return dataset_current


# Function for opening the forex website
def open_forex_website():
    
    # Open Chrome
    driver = webdriver.Chrome()

    # Maximize the windows
    driver.maximize_window()

    # Go the website
    driver.get('https://www.forex.com/en/account-login/') 
    
    # Report
    print("1. Forex.com opened.")
    
    # Wait 60 seconds 
    time.sleep(60)
    
    # Add username
    usernameElement = driver.find_element_by_name("Username")
    usernameElement.clear()
    usernameElement.send_keys("YourUsername")

    # Add password
    passwordElement = driver.find_element_by_name("Password")
    passwordElement.clear()
    passwordElement.send_keys("YourPassword")

    # Enter
    passwordElement.send_keys(Keys.RETURN)
    
    # Report
    print("2. Signed in to the website.")
    
    # Wait 120 seconds 
    time.sleep(120)
    
    # Minimize the notification
    driver.find_elements_by_class_name("tip__back-later")[0].click()

    # Click on the FX tab
    driver.find_elements_by_class_name("markets-tags__item")[2].click()

    # Click on the Major FX tab
    driver.find_elements(by = "class name", value = "markets-filter__tab")[1].click() 
    print("3. Opened FX-Major tab.")
        
    return driver


# Function for buying or selling
def buy_sell_forex(driver, action, ticker_name = "USD/JPY", quantity = 1000):
    
    # Get the index for the ticker name
    index = ["AUD/USD", "NZD/USD", "USD/CAD", "EUR/JPY", "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"].index(ticker_name)
    
    # If buying
    if (action == "buy"):
        
        # Add quantity for given ticker name
        quantityElement = driver.find_elements(by = "class name", value = "quantity__field")[index].find_elements_by_tag_name("input")[0]
        quantityElement.clear()
        quantityElement.send_keys(quantity);

        # Click for buying
        buyButtonElement = driver.find_elements_by_class_name("price--buy")[index]
        buyButtonElement.click()
    
    # If selling
    if (action == "sell"):
        
        # Add quantity for given ticker name
        quantityElement = driver.find_elements(by = "class name", value = "quantity__field")[index].find_elements_by_tag_name("input")[0]
        quantityElement.clear()
        quantityElement.send_keys(quantity);

        # Click for selling
        sellButtonElement = driver.find_elements_by_class_name("price--sell")[index]
        sellButtonElement.click()
