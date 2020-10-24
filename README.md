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





