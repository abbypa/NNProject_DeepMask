__author__ = 'Abigail'
import theano.tensor as T
import lasagne

l_in = lasagne.layers.InputLayer((100, 50))
l_hidden = lasagne.layers.DenseLayer(l_in, num_units=200)
l_out = lasagne.layers.DenseLayer(l_hidden, num_units=10, nonlinearity=T.nnet.softmax)

print("success")
