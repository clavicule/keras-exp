import argparse
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler


def baseline_model(batch_size, timesteps, data_dim):
    m = Sequential()
    m.add(LSTM(10, input_shape=(timesteps, data_dim), return_sequences=False)) #consume_less='gpu'
    m.add(Dense(1, input_dim=10))
    m.compile(loss='mse', optimizer='adam')
    return m

def load_data(X, Y, sequence_size, time_length = 24, step = 10):
    total_size = X.shape[0]
    if total_size != Y.shape[0]:
        print('X and Y do not have the same dimension')
        return
    print(total_size/sequence_size)
    print(sequence_size/step)
    X_fmt = np.zeros((total_size/sequence_size * sequence_size/step, time_length, 2), dtype="float")
    Y_fmt = np.zeros((total_size/sequence_size * sequence_size/step, 1), dtype="float")
    actual_size=0
    for i in range(0, total_size, sequence_size):
        X_batch = X[i:i+sequence_size, :]
        Y_batch = Y[i:i+sequence_size]
        for k in range(0, sequence_size, step):
            timestep_end = k + time_length
            if timestep_end + 1 < sequence_size:
                X_fmt[actual_size] = X_batch[k:timestep_end, :]
                Y_fmt[actual_size] = Y_batch[timestep_end + 1]
                actual_size += 1
    X_fmt=X_fmt[:actual_size]
    Y_fmt=Y_fmt[:actual_size]
    return X_fmt, Y_fmt

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True, help="Path to the training data file")
ap.add_argument("-v", "--validating", required=True, help="Path to the validating data file")
args = vars(ap.parse_args())

print('loading training data...')
train_dataset = np.loadtxt(args['training'], delimiter=',')
train_end = train_dataset.shape[1] - 1
# X_train = train_dataset[:, 0:train_end].astype(float)
X_train = train_dataset[:, [0, train_end]].astype(float)
Y_train = train_dataset[:, train_end].astype(float)

print('scaling data')
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)

'''
print('loading validating data...')
val_dataset = np.loadtxt(args['validating'], delimiter=',')
val_end = val_dataset.shape[1] - 1
# X_val = val_dataset[:, 0:val_end-1].astype(float)
X_val = val_dataset[:, 0].astype(float)
Y_val = val_dataset[:, val_end-1].astype(float)
K_val = val_dataset[:, val_end].astype(int)
'''

# Reshape  dimensions of input and output to match  LSTM requirements
# Input:(nb_samples, timesteps, input_dim).
# Output: if return_sequences: 3D tensor with shape: (nb_samples, timesteps, output_dim).
#         else: 2D tensor with shape: (nb_samples, output_dim).
# input ==> ( xx, 24, 2 )
# output ==> ( xx, 1 )

print('format training data')
X, Y = load_data(X_train, Y_train, 3024)
print(X.shape)
print(Y.shape)
batch_size = 25

print('building RNN')
model = baseline_model(batch_size=batch_size, timesteps=24, data_dim=2)

print('training RNN')
model.fit(x=X, y=Y, validation_split=0.2, nb_epoch=10, batch_size=batch_size)
