from datetime import datetime
from yahoo_finance_data import download_finance_data
from dateutil.relativedelta import relativedelta
from pre_processing import PreProcessing
from time2vec_transformer_model import Time2VecTransformer
from train_evaluate_model_helper import *

to_date = datetime.now()
from_date = to_date - relativedelta(days=700)
data = PreProcessing('BTC-GBP',download_finance_data(['BTC-GBP'],from_date,to_date))

model_name = 'pctg_change_7d_ma_1.1m_params'
seq_len = 14
d_k = 512
d_v = 512
n_heads = 24
ff_dim = 512
epochs = 1 #40
batch_size = 32
no_features = 5

data.apply_n_day_rolling_average(7)\
.calculate_normalised_percentage_change()

tvt = Time2VecTransformer(model_name,seq_len,d_k,d_v,n_heads,ff_dim,no_features)

md = split_train_evalute_model(data,tvt,epochs,batch_size,plot=False)
