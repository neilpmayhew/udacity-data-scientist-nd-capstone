import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(".")

import json
import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Line
from joblib import load
from sqlalchemy import create_engine
import tensorflow as tf
from model.time2vec_transformer_model import *
from model.train_evaluate_model_helper import forecast

app = Flask(__name__)

model_name = 'pctg_change_7d_ma_1.1m_params'

# load model_data
with open('pctg_change_7d_ma_1.1m_params_model_data.pkl','rb') as f:
    model_data = load(f)

# load model
model = tf.keras.models.load_model(f'{model_name}.hdf5',
                                   custom_objects={'Time2Vector': Time2Vector, 
                                                   'SingleAttention': SingleAttention,
                                                   'MultiAttention': MultiAttention,
                                                   'TransformerEncoder': TransformerEncoder})


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    train_pred = model_data.train_pred.reshape(model_data.train_pred.shape[0])
    val_pred = model_data.val_pred.reshape(model_data.val_pred.shape[0])
    test_pred = model_data.test_pred.reshape(model_data.test_pred.shape[0])
    graphs = [
        {
            'data': [
                Line(
                    x=np.arange(model_data.pre_processing.train_data.shape[0]),
                    y=model_data.pre_processing.df_train['close'],
                    name='Training'),
                Line(
                    x=np.arange(model_data.pre_processing.train_data.shape[0],model_data.pre_processing.train_data.shape[0]+model_data.pre_processing.val_data.shape[0]),
                    y=model_data.pre_processing.df_val['close'],
                    name='Validation'),
                Line(
                    x=np.arange(model_data.pre_processing.train_data.shape[0]+model_data.pre_processing.val_data.shape[0],
                                model_data.pre_processing.train_data.shape[0]+model_data.pre_processing.val_data.shape[0]+model_data.pre_processing.test_data.shape[0]),
                    y=model_data.pre_processing.df_test['close'],
                    name='Test')
            ],

            'layout': {
                'title': 'Train / Val / Test Split',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }
        },
        {
            'data':[
                Line(
                    x=np.arange(model_data.seq_len,model_data.train_pred.shape[0]+model_data.seq_len),
                    y=train_pred,
                    width=3,
                    name='Predicted BTC-GBP Closing Returns'
                )
                ,Line(y=model_data.pre_processing.train_data[:, 3],
                      name='Actual BTC-GBP Closing Returns')],
            'layout': {
                'title': 'Actual vs Predicted:- Training Data',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        },
        {
            'data':[
                Line(
                    x=np.arange(model_data.seq_len,model_data.val_pred.shape[0]+model_data.seq_len),
                    y=val_pred,
                    width=3,
                    name='Predicted BTC-GBP Closing Returns'
                )
                ,Line(y=model_data.pre_processing.val_data[:, 3],
                      name='Actual BTC-GBP Closing Returns')],
            'layout': {
                'title': 'Actual vs Predicted:- Validation Data',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        },
        {
            'data':[
                Line(
                    x=np.arange(model_data.seq_len,model_data.test_pred.shape[0]+model_data.seq_len),
                    y=test_pred,
                    width=3,
                    name='Predicted BTC-GBP Closing Returns'
                )
                ,Line(y=model_data.pre_processing.test_data[:, 3],
                      name='Actual BTC-GBP Closing Returns')],
            'layout': {
                'title': 'Actual vs Predicted:- Test Data',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        },
        {
            'data':[
                Line(
                    y=model_data.training_history['loss'],
                    width=3,
                    name='Training Loss (MSE)'
                )
                ,Line(
                    y=model_data.training_history['val_loss'],
                    name='Validation Loss (MSE)')],
            'layout': {
                'title': 'Training History:- Loss',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        },
        {
            'data':[
                Line(
                    y=model_data.training_history['mae'],
                    width=3,
                    name='Training MAE'
                )
                ,Line(
                    y=model_data.training_history['val_mae'],
                    name='Validation MAE')],
            'layout': {
                'title': 'Training History:- MAE',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        },
        {
            'data':[
                Line(
                    y=model_data.training_history['mape'],
                    width=3,
                    name='Training MAPE'
                )
                ,Line(
                    y=model_data.training_history['val_mape'],
                    name='Validation MAPE')],
            'layout': {
                'title': 'Training History:- MAPE',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }


        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON,train_eval=model_data.train_eval,val_eval=model_data.val_eval,test_eval=model_data.test_eval)


# web page to show forecast for next 7 days
@app.route('/forecast')
def forecast_route():
    forecast_date = []
    forecast_value = []

    forecast_date.append(pd.Timestamp(np.datetime64(model_data.pre_processing.df.iloc[-1,0],'D')+1))
    forecast_value.append(forecast(model_data.pre_processing.df,model_data.seq_len,model))

    graphs = [
        {
            'data': [
                Line(
                    x=model_data.pre_processing.df['date'],
                    y=model_data.pre_processing.df['close'],
                    name='BTC-GBP Actual'),
                Line(
                    x=forecast_date,
                    y=forecast_value,
                    name='BTC-GBP Forecast',
                    width=4)
            ],
            'layout': {
                'title': 'BTC-GBP Forecast',
                'yaxis': {
                    'title': "BTC Close Returns (7d ma)"
                },
                'xaxis': {
                    'title': "Day"
                }
            }
        }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
