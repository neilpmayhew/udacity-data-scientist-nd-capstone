import re
import datetime
import os
import pandas as pd
import pywt


def download_finance_data(currency_pairs, from_date, to_date):
    """
    download currency data from yahoo finance for a currency code pair between from_date and to_date.

    args:
        currency_pair:
            currency paid i.e. BTC-USD or ETH-USD
        from_date:
            date range from
        to_date:
            date range to
    returns:
        pandas data frame
    """
    period_1 = datetime.datetime.timestamp(from_date)
    period_2 = datetime.datetime.timestamp(to_date)

    df = None

    for currency_pair in currency_pairs:

        url = __get_url(currency_pair, period_1, period_2)

        _df = pd.read_csv(url)

        _df.columns = map(str.lower, _df.columns)
        _df.insert(0,'currency_pair',f'{currency_pair}')

        if df is None:
            df = _df
        else:
            df = pd.concat([df,_df],)


    return df

def __get_url(currency_pair, period_1, period_2):
    """
    Returns the url for downloading the historical currency data e.g.
    'https://query1.finance.yahoo.com/v7/finance/download/BTC-USD?period1=1538697600&period2=1601856000&interval=1d&events=history&includeAdjustedClose=true'

    args:
        currency_pair:
            the currency pair to retrieve e.g. BTC-USD
        period_1:
            from timestamp can be generated from a datetime object then converted using datetime.datetime.timestamp()
        period_2:
            to timestamp can be generated from a datetime object then converted using datetime.datetime.timestamp()
    """

    period_1 = int(period_1)
    period_2 = int(period_2)

    return f'https://query1.finance.yahoo.com/v7/finance/download/{currency_pair}?period1={period_1}&period2={period_2}&interval=1d&events=history&includeAdjustedClose=true'
