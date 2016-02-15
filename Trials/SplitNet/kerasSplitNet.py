from keras.models import *
from keras.layers.core import *

# graph model with one input and two outputs
graph = Graph()
graph.add_input(name='input', input_shape=(4,))
graph.add_node(Dense(4), name='dense1', input='input')
graph.add_node(Dense(2), name='dense2', input='dense1')
graph.add_node(Dense(2), name='dense3', input='dense1')
graph.add_output(name='output1', input='dense2')
graph.add_output(name='output2', input='dense3')

graph.compile(optimizer='sgd', loss={'output1':'mse', 'output2':'mse'})

X_train = np.array([[1,1,1,1],[0,0,0,0]])
y_train = np.array([[1,1],[0,0]])
y2_train = np.array([[0,0],[1,1]])

history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=1000, verbose=2)

X_check = np.array([[1,1,1,1],[0,0,0,0]])
res = graph.predict({'input':X_check}, verbose=2)
print res

print "DONE"
