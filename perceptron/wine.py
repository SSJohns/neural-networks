import keras as K
from keras.layers import Dense
from keras.models import Sequential
import tensorflow
import numpy as np

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

network = Sequential()
x = list()
y = list()
with open('wine.dat', 'r') as file:
	for line in file.readlines():
		if line == '\n':
			continue
		data = line.split(',')
		x.append(map(float, data[1:]))
		# y.append(data[0])
		if data[0] == '1':
			y.append([1,0,0])
		elif data[0] == '2':
			y.append([0,1,0])
		else:
			y.append([0,0,1])
x = np.array(x)
y = np.array(y)
x, y = unison_shuffled_copies(x, y)
train_x = x[0:130]
train_y = y[0:130]
test_x = x[130:]
test_y = y[130:]
# import ipdb; ipdb.set_trace()

network.add(Dense(16,input_dim=13, activation='sigmoid'))
network.add(Dense(10,activation='sigmoid'))
network.add(Dense(10, activation='sigmoid'))
network.add(Dense(3, activation='softmax'))
opt = 'Adagrad' # We'll use SDG as our optimizer
obj = 'categorical_crossentropy' # And we'll use categorical cross-entropy as the objective we're trying to minimize
network.compile(optimizer=opt, loss=obj, metrics=['accuracy']) # Include accuracy for when we want to test our net

NEPOCHS=100
# The fit method returns a history object
history = network.fit(train_x,train_y, #input, target
    nb_epoch=NEPOCHS, #number of epochs, or number of times we want to train with the entire dataset
    batch_size=1, #batch size, or number of samples trained at one time
    verbose=1) #verbosity of 1 gives us medium output

loss, acc = network.evaluate(test_x, test_y) # Returns the loss and accuracy as a tuple


print('Final accuracy after {} iterations: {}'.format(NEPOCHS, acc))

