import requests
import re
import datetime


def download_finance_data(currency_pairs, from_date, to_date,target_dir):
    """
    download currency data as a csv from yahoo finance for a currency code pair between from_date and to_date

    args:
        currency_pair:
            currency paid i.e. BTC-USD or ETH-USD
        from_date:
            date range from
        to_date:
            date range to
    """
    period_1 = datetime.datetime.timestamp(from_date)
    period_2 = datetime.datetime.timestamp(to_date)

    for currency_pair in currency_pairs:

        url = __get_url(currency_pair, period_1, period_2)

        r = requests.get(url)

        with open(
                f"{target_dir}/{currency_pair}_{from_date:%Y%m%d}_{to_date:%Y%m%d}.csv",
                'wb') as f:
            f.write(r.content)

        f.close()

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
