import argparse, os
import numpy as np
import random
import pandas as pd
import boto3
import io
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import EarlyStopping


def get_data(testcases, lookback, holdout):
    bucket = "tennessee-eastman-process-alarm-management-dataset"
    features_file_name = "data/sensors_original.csv"
    labels_file_name = "data/alarms_filtered.csv"
    s3 = boto3.client('s3') 
    print("downloading sensor data...")
    obj = s3.get_object(Bucket= bucket, Key= features_file_name) 
    features = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print("downloading alarm data...")
    obj = s3.get_object(Bucket= bucket, Key= labels_file_name) 
    labels = pd.read_csv(io.BytesIO(obj['Body'].read()))
    
    print(features.head())
    print(features.columns)
    print(labels.head())
    print(labels.columns)
    
    print("creating sequences...")
    timeseries_data = []
    for i in range(1,51):
        x = features[features['TEST_NO']==i].drop(columns=['Unnamed: 0', 'TEST_NO']).to_numpy()
        y = labels[labels['TEST_NO']==i].drop(columns=['Unnamed: 0', 'TEST_NO']).to_numpy()
        for state in range(lookback, len(x)):
            timeseries_data.append((x[state-lookback:state],y[state]))
            
    del features
    del labels
    del obj
    
    random.shuffle(timeseries_data)
    timeseries_data = random.sample(timeseries_data, testcases)
    
    print("splitting train and test...")
    x_train, x_test, y_train, y_test = train_test_split(
        list(example[0] for example in timeseries_data),
        list(example[1] for example in timeseries_data),
        test_size=holdout,
        shuffle=True
    )
    
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-count', type=int, default=0)
    parser.add_argument('--model-dir', type=str, default='/tmp')
    parser.add_argument('--lookback', type=int, default=1)
    parser.add_argument('--testcases', type=int, default=100000)
    parser.add_argument('--holdout', type=float, default=0.2)
    
    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    lr         = args.learning_rate
    batch_size = args.batch_size
    gpu_count  = args.gpu_count
    model_dir  = args.model_dir
    lookback   = args.lookback
    testcases  = args.testcases
    holdout    = args.holdout
    
    x_train, x_test, y_train, y_test = get_data(testcases, lookback, holdout)
    x_train = np.array(x_train, dtype=np.float16)
    y_train = np.array(y_train, dtype=np.int8)
    x_test = np.array(x_test, dtype=np.float16)
    y_test = np.array(y_test, dtype=np.int8)
    
    print("building model...")
    model = Sequential()
    
    #LSTM 1 returns full sequence
    model.add(
        LSTM(
            256, 
            input_shape=(lookback, 81),
            activation='tanh',
            recurrent_activation='sigmoid',
            stateful=False, 
            recurrent_dropout=0.2,
            return_sequences=True,
            recurrent_regularizer=regularizers.l2(l=0.001),
            kernel_regularizer=regularizers.l2(l=0.001),
            bias_regularizer=regularizers.l2(l=0.001)
        )
    )

    #Batch norm layer
    model.add(
        BatchNormalization
        (
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None
        )
    )

    #LSTM 2 return last output only
    model.add(
        LSTM(
            256, 
            input_shape=(81, lookback),
            activation='tanh',
            recurrent_activation='sigmoid',
            stateful=False, 
            recurrent_dropout=0.2,
            return_sequences=False,
            recurrent_regularizer=regularizers.l2(l=0.001),
            kernel_regularizer=regularizers.l2(l=0.001),
            bias_regularizer=regularizers.l2(l=0.001)
        )
    )

    #Batch norm layer
    model.add(
        BatchNormalization
        (
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None
        )
    )

    #Hidden layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))

    #Prediction layer
    model.add(Dense(81, activation='sigmoid'))

    #Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    if gpu_count > 1:
        model = multi_gpu_model(model, gpus=gpu_count)
        
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    
    print("starting training...")
    history = model.fit(
        x_train, 
        y_train, 
        validation_data=(x_test, y_test), 
        epochs=epochs, 
        batch_size=128, 
        verbose=2, 
        callbacks=[es]
    )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Validation loss    :', score[0])
    print('Validation accuracy:', score[1])
    



    
    
    
