from keras.models import Sequential
from keras.layers import Dense
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

# fit
model.fit( X, Y, nb_epoch = 150, batch_size = 10 )

# evaluate
scores = model.evaluate( X, Y )
print( '%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
