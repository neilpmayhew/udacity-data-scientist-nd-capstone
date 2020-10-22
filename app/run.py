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

app = Flask(__name__)

model_name = 'pctg_change_7d_ma_1.1m_params'

# load model_data
with open('pctg_change_7d_ma_1.1m_params_model_data.pkl','rb') as f:
    model_data = load(f'{model_name}_model_data.pkl')

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

    graphs = [
        {
            'data': [
                Line(
                    x=np.arange(model_data.pre_processing.train_data.shape[0]),
                    y=model_data.pre_processing.train_data
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
# @app.route('/go')
# def go():
#     # save user input in query
#     query = request.args.get('query', '') 

#     # use model to predict classification for query
#     classification_labels = model.predict([query])[0]
#     classification_results = dict(zip(df.columns[4:], classification_labels))

#     # This will render the go.html Please see that file. 
#     return render_template(
#         'go.html',
#         query=query,
#         classification_result=classification_results
#     )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
