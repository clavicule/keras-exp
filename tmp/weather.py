import argparse
import numpy as np
import os.path
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler


def baseline_model(batch_size, timesteps, data_dim, weight_file=''):
    m = Sequential()
    m.add(LSTM(30, input_shape=(timesteps, data_dim), return_sequences=False))  # consume_less='gpu'
    # m.add(LSTM(30, return_sequences=False))
    # m.add(LSTM(30, return_sequences=False))
    m.add(Dense(data_dim, input_dim=30))

    if os.path.isfile(weight_file):
        print('loading weights')
        model.load_weights(weight_file)

    m.compile(loss='mse', optimizer='adam')
    return m

def load_data(X, Y, time_length, step):
    total_size = X.shape[0]
    if total_size != Y.shape[0]:
        print('X and Y do not have the same dimension')
        return
    max_samples = total_size//step + 1
    X_fmt = np.zeros((max_samples, time_length, X.shape[1]), dtype="float")
    Y_fmt = np.zeros((max_samples, Y.shape[1]), dtype="float")
    actual_size=0
    for k in range(0, total_size, step):
        timestep_end = k + time_length
        if timestep_end + 1 < total_size:
            X_fmt[actual_size] = X[k:timestep_end, :]
            Y_fmt[actual_size] = Y[timestep_end + 1]
            actual_size += 1
    X_fmt = X_fmt[:actual_size]
    Y_fmt = Y_fmt[:actual_size]
    return X_fmt, Y_fmt

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="Path to the training data file")
ap.add_argument("-o", "--output", required=True, help="Path to the prediction file")
args = vars(ap.parse_args())

print('loading training data...')
train_dataset = np.loadtxt(args['training'], delimiter=',')
X, Y = load_data(train_dataset, train_dataset, time_length=40, step=10)
print(X.shape)
print(Y.shape)

print('building RNN')
batch_size = 50
checkpoint = ModelCheckpoint("rnn.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
callbacks_list = [checkpoint]
model = baseline_model(batch_size=batch_size, timesteps=X.shape[0], data_dim=X.shape[2])

print('training RNN')
model.fit(x=X, y=Y, validation_split=0.2, nb_epoch=10, batch_size=batch_size, callbacks=callbacks_list)
