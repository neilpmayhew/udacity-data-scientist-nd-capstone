import matplotlib.pyplot as plt
plt.style.use('seaborn')

import numpy as np
import pickle as pkl

class ModelData():
    def __init__(self,pre_processing,train_pred,val_pred,test_pred,train_eval,val_eval,test_eval,training_history,seq_len):
        self.pre_processing = pre_processing

        self.train_pred = train_pred
        self.val_pred = val_pred
        self.test_pred = test_pred

        self.train_eval = {'loss[mse]': train_eval[0],'mae':train_eval[1],'mape':train_eval[2]}
        self.val_eval = {'loss[mse]': val_eval[0],'mae':val_eval[1],'mape':val_eval[2]}
        self.test_eval = {'loss[mse]': test_eval[0],'mae':test_eval[1],'mape':test_eval[2]}
        self.training_history = training_history

        self.seq_len = seq_len

def forecast(df, seq_len,model):

    seq = np.array([df.iloc[-seq_len:,-5:].values])

    return model.predict(seq)[0][0]

def split_train_evalute_model(pre_processing,model,epochs,batch_size,plot=True):

    model.model.summary()

    print('-------------------------------')
    print('Performing Train/Val/Test split')
    print('-------------------------------')

    pre_processing.generate_train_val_test_split(model.seq_len,'close')

    if plot:
        plot_train_val_test_split(pre_processing)

    print('-------------------------------')
    print(f'Fitting model: {model.model_name}')
    print('-------------------------------')

    model.fit(pre_processing.X_train,pre_processing.y_train,pre_processing.X_val,pre_processing.y_val,batch_size,epochs)

    print('------------------------------------------')
    print(f'Generating predictions: {model.model_name}')
    print('------------------------------------------')

    #Calculate predication for training, validation and test data
    train_pred = model.best_model.predict(pre_processing.X_train)
    val_pred = model.best_model.predict(pre_processing.X_val)
    test_pred = model.best_model.predict(pre_processing.X_test)

    #Print evaluation metrics for all datasets
    train_eval = model.best_model.evaluate(pre_processing.X_train, pre_processing.y_train, verbose=0)
    val_eval = model.best_model.evaluate(pre_processing.X_val, pre_processing.y_val, verbose=0)
    test_eval = model.best_model.evaluate(pre_processing.X_test, pre_processing.y_test, verbose=0)

    print('---------------------------------------')
    print(f"Evaluation metrics: {model.model_name}")
    print('---------------------------------------')
    print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
    print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
    print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

    if plot:
        plot_results(pre_processing,model,train_pred,val_pred,test_pred)

    if plot:
        plot_model_metrics(model)

    model_data = ModelData(pre_processing,train_pred,val_pred,test_pred,train_eval,val_eval,test_eval,model.history.history,model.seq_len)

    with open(f'{model.model_name}_model_data.pkl',"wb") as f:
         pkl.dump(model_data, f)

def plot_train_val_test_split(pre_processing):
    fig = plt.figure(figsize=(15,12))
    st = fig.suptitle("Data Separation", fontsize=20)
    st.set_y(0.95)

    ###############################################################################

    ax1 = fig.add_subplot(211)
    ax1.plot(np.arange(pre_processing.train_data.shape[0]), pre_processing.df_train['close'], label='Training data')

    ax1.plot(np.arange(pre_processing.train_data.shape[0], 
                    pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0]), pre_processing.df_val['close'], label='Validation data')

    ax1.plot(np.arange(pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0], 
                    pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0]+pre_processing.test_data.shape[0]), pre_processing.df_test['close'], label='Test data')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Closing Returns')
    ax1.set_title("Close Price", fontsize=18)
    ax1.legend(loc="best", fontsize=12)

    ###############################################################################

    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(pre_processing.train_data.shape[0]), pre_processing.df_train['volume'], label='Training data')

    ax2.plot(np.arange(pre_processing.train_data.shape[0], 
                    pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0]), pre_processing.df_val['volume'], label='Validation data')

    ax2.plot(np.arange(pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0], 
                    pre_processing.train_data.shape[0]+pre_processing.val_data.shape[0]+pre_processing.test_data.shape[0]), pre_processing.df_test['volume'], label='Test data')
    ax2.set_xlabel('date')
    ax2.set_ylabel('Normalized Volume Changes')
    ax2.set_title("volume", fontsize=18)
    ax2.legend(loc="best", fontsize=12)

    plt.show()

def plot_results(pre_processing,model,train_pred,val_pred,test_pred):
    ###############################################################################
    # Display results

    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle(f"Time2Vec Transformer Model ({model.model_name}) Results", fontsize=22)
    st.set_y(0.92)

    #Plot training data results
    ax11 = fig.add_subplot(311)
    ax11.plot(pre_processing.train_data[:, 3], label='BTC-GBP Closing Returns')
    ax11.plot(np.arange(model.seq_len, train_pred.shape[0]+model.seq_len), train_pred, linewidth=3, label='Predicted BTC-GBP Closing Returns')
    ax11.set_title("Training Data", fontsize=18)
    ax11.set_xlabel('Date')
    ax11.set_ylabel('BTC-GBP Closing Returns')
    ax11.legend(loc="best", fontsize=12)

    #Plot validation data results
    ax21 = fig.add_subplot(312)
    ax21.plot(pre_processing.val_data[:, 3], label='BTC-GBP Closing Returns')
    ax21.plot(np.arange(model.seq_len, val_pred.shape[0]+model.seq_len), val_pred, linewidth=3, label='Predicted BTC-GBP Closing Returns')
    ax21.set_title("Validation Data", fontsize=18)
    ax21.set_xlabel('Date')
    ax21.set_ylabel('BTC-GBP Closing Returns')
    ax21.legend(loc="best", fontsize=12)

    #Plot test data results
    ax31 = fig.add_subplot(313)
    ax31.plot(pre_processing.test_data[:, 3], label='BTC-GBP Closing Returns')
    ax31.plot(np.arange(model.seq_len, test_pred.shape[0]+model.seq_len), test_pred, linewidth=3, label='Predicted BTC-GBP Closing Returns')
    ax31.set_title("Test Data", fontsize=18)
    ax31.set_xlabel('Date')
    ax31.set_ylabel('BTC-GBP Closing Returns')
    ax31.legend(loc="best", fontsize=12)

    plt.show()
def plot_model_metrics(model):
    ## Display model metrics

    fig = plt.figure(figsize=(15,20))
    st = fig.suptitle(f"Time2Vec Transformer Model {model.model_name} Metrics", fontsize=22)
    st.set_y(0.92)

    #Plot model loss
    ax1 = fig.add_subplot(311)
    ax1.plot(model.history.history['loss'], label='Training loss (MSE)')
    ax1.plot(model.history.history['val_loss'], label='Validation loss (MSE)')
    ax1.set_title("Model loss", fontsize=18)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend(loc="best", fontsize=12)

    #Plot MAE
    ax2 = fig.add_subplot(312)
    ax2.plot(model.history.history['mae'], label='Training MAE')
    ax2.plot(model.history.history['val_mae'], label='Validation MAE')
    ax2.set_title("Model metric - Mean average error (MAE)", fontsize=18)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean average error (MAE)')
    ax2.legend(loc="best", fontsize=12)

    #Plot MAPE
    ax3 = fig.add_subplot(313)
    ax3.plot(model.history.history['mape'], label='Training MAPE')
    ax3.plot(model.history.history['val_mape'], label='Validation MAPE')
    ax3.set_title("Model metric - Mean average percentage error (MAPE)", fontsize=18)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Mean average percentage error (MAPE)')
    ax3.legend(loc="best", fontsize=12)

    plt.show()
