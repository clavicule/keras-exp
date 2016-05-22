from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import StratifiedKFold
import numpy as np

# init seed
seed = 7
np.random.seed(seed)

# load data (CSV)
dataset = np.loadtxt('pima-indians-diabetes.data', delimiter=',')

# split in put and output
X = dataset[:, 0:8]
Y = dataset[:, 8]

kfold = StratifiedKFold( y = Y, n_folds=10, shuffle=True, random_state=seed)
cvscores = []

for i, (train_index, test_index) in enumerate(kfold):
    # make model
    model = Sequential()
    model.add( Dense( 12, input_dim = 8, init = 'uniform', activation = 'relu' ) )
    model.add( Dense( 8, init = 'uniform', activation = 'relu' ) )
    model.add( Dense( 1, init = 'uniform', activation = 'sigmoid' ) )

    # compile model
    model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

    # fit
    model.fit(X[train_index], Y[train_index], validation_split=0.33, nb_epoch = 150, batch_size = 10, verbose=0)

    # evaluate
    scores = model.evaluate(X[test_index], Y[test_index], verbose=0)
    print( '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)

print('%.2f%% (+/- %.2f%%)' % (np.mean(cvscores), np.std(cvscores)))

