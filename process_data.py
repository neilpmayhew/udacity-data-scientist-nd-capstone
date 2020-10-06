import numpy as np
import pandas as pd

def clean_and_prep_df(csv_dictionary):
   """
   function to clean and prepare a yahoo finance data csv ready for training with pytorch-forecasting

   args:
       csv_dictionary:
           dictionary yahoo finance binary csv files. Keys are the currency key-pair contained in the file e.g. BTC-USD or ETH-USD
   """
   df_all_currencies = pd.DataFrame()

   for currency_pair,csv_path in csv_dictionary.items():

      df = pd.read_csv(csv_path)

      df['date'] = pd.to_datetime(df.date)

      df.columns = map(str.lower, df.columns)

      # add time index
      df["time_idx"] = np.argsort(df["date"])

      # store currency pair
      df["currency_pair"] = currency_pair

      # only a single day with nans so fill with value from previous day
      df = df.fillna(method='ffill')

      df['log_close'] = np.log(df['close'])
      df['month_name'] = df.date.dt.month_name()
      df['year'] = df.date.dt.year

      pd.concat(df_all_currencies,df)

   return df_all_currencies
