import numpy as np
import pandas as pd
import pywt

class PreProcessing():
   """
   Class to clean and prepare dataframe ready for training with the TimeEmbeddingTransformer network to be trained with tensorflow

   Provides a base function to parse dates to datetime type and handle nans as well as providing additional functions to engineer features. Each functions returns the class itself
   allowing the functions to be stacked.
   """
   def __init__(self,currency_pair,df):
      """
      Init class taking a pandas data frame of raw currency data

      args:
          currency_pair:
              The currency pair for the data set
          df:
              pandas data frame
      """
      self.currency_pair = currency_pair
      self.base_clean_and_prep_df(df)

   def base_clean_and_prep_df(self,df):
      """
      function to clean and prepare a yahoo finance data df ready for training

      args:
         df:
            pandas df of data set to be cleaned and prepared for training.
      returns:
         The cleaned data as a pandas data frame
      """
      df_clean = df

      df_clean['date'] = pd.to_datetime(df_clean.date)

      # only a single nan so ffil from previous day
      df_clean = df_clean.fillna(method='ffill')

      # select only required columns
      df_clean = df_clean[['date','open','high','low','close','volume']]

      # sort by date
      df_clean.sort_values('date', inplace=True)

      self.df = df_clean

   def calculate_normalised_percentage_change(self):
      """
       Converts actual OHLCV values to percentage change over prior value and then applies a Min/Max scale to normalise 
      """

      # Convert OHLCV columns to percentage change
      self.df['open'] = self.df['open'].pct_change() 
      self.df['high'] = self.df['high'].pct_change() 
      self.df['low'] = self.df['low'].pct_change()
      self.df['close'] = self.df['close'].pct_change()
      self.df['volume'] = self.df['volume'].pct_change()

      self.df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

      self.df = self.df.reset_index(drop=True)

      # # Calculate Min/Max of price columns'

      min_return = min(self.df[['open', 'high', 'low', 'close']].min(axis=0))
      max_return = max(self.df[['open', 'high', 'low', 'close']].max(axis=0))

      # # Min-max normalize price columns (0-1 range)
      self.df['open'] = (self.df['open'] - min_return) / (max_return - min_return)
      self.df['high'] = (self.df['high'] - min_return) / (max_return - min_return)
      self.df['low'] = (self.df['low'] - min_return) / (max_return - min_return)
      self.df['close'] = (self.df['close'] - min_return) / (max_return - min_return)

      # Calculate Min/Max of price columns'

      min_volume = self.df['volume'].min(axis=0)
      max_volume = self.df['volume'].max(axis=0)

      # Min-max normalize volume columns (0-1 range)
      self.df['volume'] = (self.df['volume'] - min_volume) / (max_volume - min_volume)

      return self

   def apply_wavelet_transform(self):
      self.df[['open', 'high', 'low', 'close', 'volume']].apply(lambda x: self.__apply_wavelet_transform(x),axis=1)

      return self

   def apply_n_day_rolling_average(self,n,in_place=True):

      target_columns = ['open', 'high', 'low', 'close', 'volume']

      if in_place == False:
         target_columns = [column + f'_{n}d_ma' for column in target_columns]

      self.df[target_columns] = self.df[['open', 'high', 'low', 'close', 'volume']].rolling(n).mean()

      # Drop all rows with NaN values
      self.df.dropna(how='any', axis=0, inplace=True) 

      return self
   def apply_n_day_exp_weighted_mean(self,n,in_place=True):

      target_columns = ['open', 'high', 'low', 'close', 'volume']

      if in_place == False:
         target_columns = [column + '_ewm' for column in target_columns]

      self.df[target_columns] = self.df[['open', 'high', 'low', 'close', 'volume']].ewm(span=n).mean()

      return self
   def __apply_wavelet_transform(self,x):
      (ca, cd) = pywt.dwt(x, "haar")
      cat = pywt.threshold(ca, np.std(ca), mode="soft")
      cdt = pywt.threshold(cd, np.std(cd), mode="soft")
      tx = pywt.idwt(cat, cdt, "haar")

      return tx

   def generate_train_val_test_split(self, seq_len, target):
      '''Create training, validation and test split'''
      df_split = self.df.copy()

      # drop date columns
      df_split = df_split.drop(columns=['date'])

      # Find array index where target variable is
      target_idx = np.argwhere(df_split.columns.values == 'close')

      df_train, df_val,df_test = np.split(df_split,[int(0.8*len(df_split)),int(0.9*len(df_split))])

      # split into train,val,test
      train_data, val_data,test_data = df_train.values, df_val.values, df_test.values

      # Training data
      X_train, y_train = [], []
      for i in range(seq_len, len(train_data)):
         X_train.append(train_data[i-seq_len:i])
         y_train.append(train_data[:, target_idx][i])
      X_train, y_train = np.array(X_train), np.array(y_train)

      # Validation data
      X_val, y_val = [], []
      for i in range(seq_len, len(val_data)):
         X_val.append(val_data[i-seq_len:i])
         y_val.append(val_data[:, target_idx][i])
      X_val, y_val = np.array(X_val), np.array(y_val)

      # Test data
      X_test, y_test = [], []
      for i in range(seq_len, len(test_data)):
         X_test.append(test_data[i-seq_len:i])
         y_test.append(test_data[:, target_idx][i])
      X_test, y_test = np.array(X_test), np.array(y_test)

      self.df_train = df_train
      self.df_val = df_val
      self.df_test = df_test

      self.train_data = train_data
      self.val_data = val_data
      self.test_data = test_data

      self.X_train = X_train
      self.y_train = y_train

      self.X_val = X_val
      self.y_val = y_val

      self.X_test = X_test
      self.y_test = y_test

      print('Training data shape: {}'.format(train_data.shape))
      print('Validation data shape: {}'.format(val_data.shape))
      print('Test data shape: {}'.format(test_data.shape))

      print('Training set shape', X_train.shape, y_train.shape)
      print('Validation set shape', X_val.shape, y_val.shape)
      print('Testing set shape' ,X_test.shape, y_test.shape)
