# BTC-GBP Forecast
## Udacity Data Scientist Nanodegree Project

### Project Overview
This project seeks to forecast the future value of Bit Coin in GBP, initially from historical data alone. Historically this has been done by humans analysing charts plotting various different metrics in order to look for common patterns or signals to inform when to buy and sell. In more modern times a lot of this is done algorithmically and more recently still machine learning has been employed to assist with this.

Financial data for Bit Coin is time series data. This project will make use of data from the Yahoo Finance service. For a particular currency pair e.g. BTC-GBP (Bit Coin to Great British Pound) a time series date set of daily values can be downloaded for Open, High, Low, Close and Volume.

The project will seek to forecast the Close value.

### Project Statement
The project needed to accomplish the following:

1. Download a data set from Yahoo Finance for a date range
2. Clear and Pre-Process the data ready
3. Train and validate the model
4. Create a Web App to:
    a. Display metrics and visualise these metrics so the user can judge how well the model has fit the data
    b. Allow the user to forecast what will happen tomorrow. This is by no means expected to be an exact forecast as this is likely impossible to do. The expectation is to forecast a value to give an idea ot the general tendency given historical data
    
### Metrics
This will be modelled as a regression problem and the loss function chosen will be mean squared error (MSE). Additionally mean absolute error (MAE) and mean absolute percentage error (MAPE) will also be calculated and used to score the model.


### Files
./data/
    yahoo_finance_data.py - class to download finance data from the yahoo web service for a currency pair e.g. BTC-GBP and for a date range
    pre_processing.py - class to cleanse and pre-process data downloaded from yahoo finance ready for the machine learning model. Contains functions to engineer features which can be combined or used to replace existing data as required.

./model/
    time2vec_transformer_model.py - class to create the tensorflow transformer model ready for training with keras
    train_evaluate_model_helper.py - helper class to automate training and evaluating the model. Can be used with a notebook for experimenting with features and tuning or used in scheduled script to produce a daily update model and data set viewable on the web append
    build_model.py - Script to be scheduled to run daily to build model and save  data needed for the web append
    
./notebook/ 
    btc-gbp-data-eda.ipynb - jupyter notebook conducting exploratory data analysis on the data set to check which cleansing and pre-processing steps will be needed. Data and potential features are calculated and visualised.
    build_test_model.ipynb - jupyter notebook to experiment with different feature and to tune the model. Training result metrics and visualations are displayed
    
./app/ 
    run.py - main script for the flask append
    templates/master.html - the main template used to display training metrics and visualations and provide forecast function
    
### How to run
To run the flask web app:

1. Run the build model script `python ./model/build_model.py` to build, train and save model and data needed for the web app
2. Run the web app `python ./app/run.py`. The console will return the URL you need to visit to use the web application


### Conclusion

    

