import argparse
import numpy as np
import os.path
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from progressbar import ProgressBar


def baseline_model(timesteps, data_dim, weight_file=''):
    m = Sequential()
    m.add(LSTM(30, input_shape=(timesteps, data_dim), return_sequences=False))  # consume_less='gpu'
    # m.add(LSTM(30, return_sequences=False))
    # m.add(LSTM(30, return_sequences=False))
    m.add(Dense(data_dim, input_dim=30))

    if os.path.isfile(weight_file):
        print('loading weights')
        m.load_weights(weight_file)

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
trainset = np.loadtxt(args['training'], delimiter=',')
Y = trainset

print('scaling data')
ss = StandardScaler()
ss.fit(trainset)
X = ss.transform(trainset)

print('reformat data')
timelength = 40
nb_step = 3
X, Y = load_data(X, Y, time_length=timelength, step=nb_step)
print(X.shape)
print(Y.shape)

print('building RNN')
batch_size = 30
dim=11
checkpoint = ModelCheckpoint("rnn.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
callbacks_list = [checkpoint]
model = baseline_model(timelength, data_dim=dim)

print('training RNN')
model.fit(x=X, y=Y, validation_split=0.05, nb_epoch=2, batch_size=batch_size, callbacks=callbacks_list)

print('predicting data')
model = baseline_model(timelength, data_dim=dim, weight_file='rnn.hdf5')

prediction_timestep = 12 * 144
val_prediction = np.zeros((prediction_timestep, dim), dtype='float')
input = ss.transform(trainset[trainset.shape[0] - timelength : trainset.shape[0]])
# input = X[X.shape[0]-1]

pbar = ProgressBar(maxval=prediction_timestep).start()
for t in range(prediction_timestep):
    output = model.predict(input)
    val_prediction[t] = output
    input = np.roll(input, -1, axis=0)
    input[timelength-1] = ss.transform(output)
    pbar.update(progress)
    progress += 1
pbar.finish()

print('val prediction shape = {}'.format(val_prediction.shape))
np.savetxt(val_prediction, delimiter=',')
