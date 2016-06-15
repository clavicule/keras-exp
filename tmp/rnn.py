import argparse
import numpy as np
import os.path
# from progressbar import ProgressBar
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def baseline_model(batch_size, timesteps, data_dim, output_dim, weight_file=''):
    m = Sequential()
    m.add(LSTM(200, input_shape=(timesteps, data_dim), return_sequences=True))  # consume_less='gpu'
    m.add(LSTM(100, return_sequences=True))
    m.add(LSTM(30, return_sequences=False))
    m.add(Dense(output_dim, input_dim=30))

    if os.path.isfile(weight_file):
        print('loading weights')
        m.load_weights(weight_file)

    m.compile(loss='mse', optimizer='adam')
    return m


def load_data(X, Y, sequence_size, time_length, step):
    total_size = X.shape[0]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    if total_size != Y.shape[0]:
        print('X and Y do not have the same dimension')
        return
    max_samples = total_size//step + 1
    X_fmt = np.zeros((max_samples, time_length, input_dim), dtype="float")
    Y_fmt = np.zeros((max_samples, output_dim), dtype="float")
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
ap.add_argument("-o", "--output", required=True, help="Path to the validating data file")
args = vars(ap.parse_args())

nb_district = 66
batch_size = 100
timestep = 40
step = 5
prediction_timestep = 1296

print('loading training data...')
train_dataset = np.loadtxt(args['training'], delimiter=',')
train_end = train_dataset.shape[1] - 1
nb_poi = train_end - nb_district - 2
# col_idx = np.arange(nb_poi + 1) + 1
# col_idx[nb_poi] = train_end - 1
# col idx so that input are POI, $ and destination districts
X_train = train_dataset[:, 1:train_end - 1].astype(float) # -2 applied
# output is destination district only
Y_train = train_dataset[:, nb_poi + 1:train_end - 1].astype(float)  # -2 applied

input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]
print('# POI = {}'.format(nb_poi))
print('input dim = {}'.format(input_dim))
print('output dim = {}'.format(output_dim))


# print('scaling data')
# ss = StandardScaler()
# ss.fit(X_train)
# X_train_normed = ss.transform(X_train)

# Reshape  dimensions of input and output to match LSTM requirements
# Input:(nb_samples, timesteps, input_dim).
# Output: if return_sequences: 3D tensor with shape: (nb_samples, timesteps, output_dim).
#         else: 2D tensor with shape: (nb_samples, output_dim).
# input ==> ( xx, 24, 2 )
# output ==> ( xx, 1 )


print('format training data')
# X, Y = load_data(X_train_normed, Y_train, sequence_size=3024, time_length=24, step=10)
X, Y = load_data(X_train, Y_train, sequence_size=3024, time_length=timestep, step=step)
print(X.shape)
print(Y.shape)

print('preparing initial data for validation')
# sample_by_district = X.shape[0] // nb_district
sample_by_district = 3024
print('# samples by district = {}'.format(sample_by_district))
init_input = np.zeros((nb_district, timestep, input_dim), dtype='float')
for d in range(nb_district):
    # get the last X-timesteps destination district for each starting district
    init_input[d] = X_train[(d+1)*sample_by_district-timestep:(d+1)*sample_by_district, :]

print('loading validation data...')
val_dataset = np.loadtxt(args['validating'], delimiter=',')
val_end = val_dataset.shape[1]
XY_val = np.zeros((nb_district, prediction_timestep, input_dim + 1), dtype='float')
for d in range(nb_district):
    XY_val[d] = val_dataset[d*prediction_timestep:(d+1)*prediction_timestep, 1:val_end - 2].astype(float) # -2 applied

print('building RNN')
checkpoint = ModelCheckpoint("output/rnn.hdf5", verbose=1, save_best_only=True, monitor='val_loss')
callbacks_list = [checkpoint]
model = baseline_model(batch_size=batch_size, timesteps=timestep, data_dim=input_dim, output_dim=output_dim)

# print('training RNN')
model.fit(x=X, y=Y, validation_split=0.2, nb_epoch=3, batch_size=batch_size, callbacks=callbacks_list)
# model.fit(x=X, y=Y, validation_split=0.2, nb_epoch=10, batch_size=batch_size, callbacks=callbacks_list)

model = baseline_model(batch_size=batch_size, timesteps=timestep, data_dim=input_dim, output_dim=output_dim, weight_file='output/rnn.hdf5')

print('predicting data')
val_prediction = np.zeros((nb_district, prediction_timestep, output_dim), dtype='float')
# pbar = ProgressBar(maxval=nb_district*prediction_timestep).start()
# progress=1
for t in range(prediction_timestep):
    if t % 300 == 0:
        print('time {}'.format(t))
    # input = np.zeros((nb_district, prediction_timestep, input_dim), dtype='float')
    input = init_input
    # input[0] = ss.transform(input[0])
    output = np.round(model.predict(input))
    val_prediction[:, t] = output
    init_input = np.roll(init_input, -1, axis=1)
    if XY_val[0][t][input_dim - 1] == 0:
        init_input[:, timestep-1, nb_poi:input_dim + 1] = output
    else:
        init_input[:, timestep-1, nb_poi:input_dim + 1] = XY_val[:, t, 1 + nb_poi:input_dim + 1]
    # pbar.update(progress)
    # progress += 1
# pbar.finish()
print('val prediction shape = {}'.format(val_prediction.shape))

print('calculating mse')
mape=0
counter = 0
# pbar = ProgressBar(widgets=[SimpleProgress()], maxval=nb_district*prediction_timestep).start()
output = open(args['output'], 'w')
# output2 = open('bis_' + args['output'], 'w')
for d in range(nb_district):
    for t in range(prediction_timestep):
        output.write(str(d+1) + ',' + '2016-01-' + str(22 + t // 144) + '-' + str(t % 144))
        for o in range(output_dim):
            output.write(',' + str(val_prediction[d][t][o]))
        output.write('\n')
        if XY_val[d][t][input_dim - 1] != 0:
            print('MSE = {}'.format(mean_squared_error(XY_val[d, t, nb_poi:input_dim], val_prediction[d, t, :])))
            # mape += abs(XY_val[d][t][input_dim - 2] - val_prediction[d][t][output_dim-1]) / XY_val[d][t][input_dim - 2]
            # counter += 1
            # output2.write(str(d + 1) + ',' + '2016-01-' + str(22 + t // 144) + '-' + str(t % 144) + ',' +
            #              str(val_prediction[d][t][output_dim - 2]) + '\n')
            # pbar.update(counter)
# pbar.finish()
output.close()
# output2.close()
# print('MAPE = {}'.format(mape/counter))
