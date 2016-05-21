from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
import numpy as np

# init seed
seed = 7
np.random.seed(seed)

# load data (CSV)
dataset = np.loadtxt('pima-indians-diabetes.data', delimiter=',')

# split in put and output
X = dataset[:, 0:8]
Y = dataset[:, 8]

# make model
model = Sequential()
model.add( Dense( 12, input_dim = 8, init = 'uniform', activation = 'relu' ) )
model.add( Dense( 8, init = 'uniform', activation = 'relu' ) )
model.add( Dense( 1, init = 'uniform', activation = 'sigmoid' ) )

# compile model
model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'] )

# fit with auto validation
#model.fit( X, Y, validation_split=0.33, nb_epoch = 150, batch_size = 10 )

# fit with manual validation
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.33, random_state=seed)
model.fit( X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch = 150, batch_size = 10 )


# evaluate
scores = model.evaluate( X, Y )
print( '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
