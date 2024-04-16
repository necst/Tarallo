import os
import pandas as pd
import re
import hashlib
import json
import time

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split

import keras
from keras import Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Embedding, Conv1D, Conv2D, Multiply, GlobalMaxPooling1D, Dropout, Activation, RNN, LSTM, Bidirectional
from keras.layers import UpSampling2D, Flatten, merge, MaxPooling2D, MaxPooling1D, UpSampling1D, AveragePooling1D, GlobalMaxPooling2D
from keras.models import load_model, Model
from keras.layers import merge, Dropout, BatchNormalization, Maximum, Add, Lambda, Concatenate
from sklearn.model_selection import train_test_split
from keras import backend as K
# from tensorflow.keras.layers import Layer
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss, confusion_matrix, roc_curve, auc

from config.config_paths import general_paths

"""
The original authors of the model asked us to not share the code, so we are not able to provide the full code.
"""

global DATASET_PATH

treshold = [0.36068833, 0.9682689, 0.3408626, 0.81711864]
idx = 1 # best model index
batch_size = 256
max_length = 1000

class ClassifyGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=batch_size, dim=max_length, shuffle=False):
        'Initialization'
        # Redacted

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Redacted

        return X, y

    def on_epoch_end(self):
        # Redacted

    def __data_generation(self, list_IDs_temp):
        # Redacted
        return X, y

class Model():
    def __init__(self):
        self.start_time = time.time()

    def get_model(self, model_path = None):  
        # Redacted
        return model

    def train(self, max_epoch, batch_size, x_train, y_train, x_val, y_val, x_test, y_test):
       # Redacted
        return model

def predict(model_name, data, label):
    # Redacted
    return y_pred


def create_results(file_names, y_pred):
    # Redacted
    return results


def write_results(results, output_path):
    with open(output_path, 'w') as f:
        json.dump(results, f)

def evaluate_zhang(input_path, output_path):
    global DATASET_PATH
    DATASET_PATH = input_path # path to the dataset

    file_names = [ e[:-4] for e in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, e)) and e.endswith('.npy')]
    file_names = [d for d in file_names if np.load(os.path.join(DATASET_PATH, d + '.npy'), allow_pickle=True).shape != ()]
    # This wants an array of filenames that will be searched for at DATASET_PATH, with shape (N,1)
    file_names_pd = pd.DataFrame(file_names, columns=['file_name']) 
    model_name = os.path.join(general_paths['zhang_models_path'], "cnn_lstm_model_" + str(idx) + ".h5")
    # BEWARE
    # Assumes that we are using only malware (that is, the label is 1) - np.ones((len_input, 1), dtype=int)
    y_pred = predict(model_name, file_names_pd, np.ones((file_names_pd.shape[0], 1), dtype=int) )
    y_pred = np.where(y_pred > treshold[idx], 1, 0)

    results = create_results(file_names, y_pred)
    write_results(results, output_path)



if __name__ == '__main__':
    data = [ e[:-4] for e in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, e)) and e.endswith('.npy')]
    data = [d for d in data if np.load(os.path.join(DATASET_PATH, d + '.npy'), allow_pickle=True).shape != ()]
    # This wants an array of filenames that will be searched for at DATASET_PATH, with shape (N,1)
    x_test = pd.DataFrame(data, columns=['file_name']) 

    # print(x_test)

    len_input = x_test.shape[0]

    # Just set everything to 1 (malware)
    y_test = np.ones((len_input,1), dtype=int)

    stacked_array = np.empty((len_input, 0))

    model_indexes = [0, 1, 2, 3]
    for idx in model_indexes:
        model_name = os.path.join(MODELS_PATH, "cnn_lstm_model_" + str(idx) + ".h5")

        # BEWARE
        # Assumes that we are using only malware (that is, the label is 1) - np.ones((len_input, 1), dtype=int)
        y_pred = predict(model_name, x_test, np.ones((len_input, 1), dtype=int) )
        
        if idx == 0:
            m1 = y_pred
        elif idx == 1:
            m2 = y_pred
        elif idx == 2:
            m3 = y_pred
        elif idx == 3:
            m4 = y_pred
        
        
    print("Model 1: ", evaluate(m1, y_test))
    print("Model 2: ", evaluate(m2, y_test))
    print("Model 3: ", evaluate(m3, y_test))
    print("Model 4: ", evaluate(m4, y_test))
