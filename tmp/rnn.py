import argparse
import numpy as np
# from progressbar import ProgressBar
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from scipy.ndimage.interpolation import shift
from sklearn.preprocessing import StandardScaler


def baseline_model(batch_size, timesteps, data_dim):
    m = Sequential()
    m.add(LSTM(30, input_shape=(timesteps, data_dim), return_sequences=False))  # consume_less='gpu'
    # m.add(LSTM(30, return_sequences=False))
    # m.add(LSTM(30, return_sequences=False))
    m.add(Dense(1, input_dim=30))
    m.compile(loss='mse', optimizer='adam')
    return m

def load_data(X, Y, sequence_size, time_length, step):
    total_size = X.shape[0]
    if total_size != Y.shape[0]:
        print('X and Y do not have the same dimension')
        return
    max_samples = total_size//step + 1
    X_fmt = np.zeros((max_samples, time_length, 2), dtype="float")
    Y_fmt = np.zeros((max_samples, 1), dtype="float")
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
X_train = train_dataset[:, [0, train_end]].astype(float)
Y_train = train_dataset[:, train_end].astype(float)

print('scaling data')
ss = StandardScaler()
ss.fit(X_train)
X_train = ss.transform(X_train)

# Reshape  dimensions of input and output to match  LSTM requirements
# Input:(nb_samples, timesteps, input_dim).
# Output: if return_sequences: 3D tensor with shape: (nb_samples, timesteps, output_dim).
#         else: 2D tensor with shape: (nb_samples, output_dim).
# input ==> ( xx, 24, 2 )
# output ==> ( xx, 1 )

print('format training data')
X, Y = load_data(X_train, Y_train, sequence_size=3024, time_length=24, step=10)
print(X.shape)
print(Y.shape)

batch_size = 300
timestep=24
step=10

print('preparing initial data for validation')
nb_district = 66
sample_by_district = X.shape[0] // nb_district
print('# samples by district = {}'.format(sample_by_district))
init_input = np.zeros((nb_district, timestep), dtype='float')
for d in range(nb_district):
    init_input[d] = X_train[(d+1)*sample_by_district-timestep:(d+1)*sample_by_district,1]

print('building RNN')
checkpoint = ModelCheckpoint("output/rnn.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
callbacks_list = [checkpoint]
model = baseline_model(batch_size=batch_size, timesteps=timestep, data_dim=2)

print('training RNN')
model.fit(x=X, y=Y, validation_split=0.2, nb_epoch=3, batch_size=batch_size, callbacks=callbacks_list)

print('predicting data')
val_prediction = np.zeros((nb_district,1440), dtype='float')
# pbar = ProgressBar(maxval=nb_district*1440).start()
progress=1
for d in range(nb_district):
    print('district {}'.format(d))
    for t in range(1440):
        if t % 300 == 0:
            print('time {}'.format(t))
        input = np.zeros((1, timestep, 2), dtype='float')
        input[0, :, 0] = d
        input[0, :, 1] = init_input[d]
        output = model.predict(input)
        val_prediction[d][t] = output
        init_input = shift(init_input, -1, cval=output)
        # pbar.update(progress)
        progress += 1
# pbar.finish()
print('val prediction shape = {}'.format(val_prediction.shape))

print('loading validation data...')
val_dataset = np.loadtxt(args['validating'], delimiter=',')
val_end = val_dataset.shape[1] - 1
XY_val = np.zeros((nb_district, 1440, 2), dtype='float')
for d in range(nb_district):
    XY_val[d] = val_dataset[d*1440:(d+1)*1440, [val_end-1, val_end]].astype(float)

print('calculating mape')
mape=0
counter = 0
# pbar = ProgressBar(widgets=[SimpleProgress()], maxval=nb_district*1440).start()
for d in range(nb_district):
    print('district {}'.format(d))
    for t in range(1440):
        if t % 300 == 0:
            print('time {}'.format(t))
        if XY_val[d][t][0] != 0. and XY_val[d][t][1] != 0:
            mape += abs(XY_val[d][t][0] - val_prediction[d][t]) / XY_val[d][t][0]
            counter += 1
            # pbar.update(counter)
# pbar.finish()
print('MAPE = {}'.format(mape/counter))
