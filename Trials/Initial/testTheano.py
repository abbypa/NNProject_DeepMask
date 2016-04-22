import theano
import theano.tensor as T
import theano.tensor.nnet as nnet
import numpy as np

# simple example-
x = T.dscalar()
fx = T.exp(T.sin(x**2))
f = theano.function(inputs=[x], outputs=[fx])
fp = T.grad(fx, wrt=x)
fprime = theano.function([x], fp)

# simple XOR net
x = T.dvector()
y = T.dscalar()


def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x)
    h = nnet.sigmoid(m)
    return h


def grad_desc(cost, theta):
    alpha = 0.1  # learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))

# Since our weight matrices will take on definite values, they're not going to be represented as Theano variables.
# A shared variable is what we use for things we want to give a definite value but we also want to update.
# Notice that I didn't define the alpha or b (the bias term) as shared variables, I just hard-coded them as strict
# values because I am never going to update/modify them.

theta1 = theano.shared(np.array(np.random.rand(3, 3), dtype=theano.config.floatX))
theta2 = theano.shared(np.array(np.random.rand(4, 1), dtype=theano.config.floatX))

hid1 = layer(x, theta1)  # hidden layer
out1 = T.sum(layer(hid1, theta2))  # output layer
fc = (out1 - y) ** 2  # cost expression

cost = theano.function(inputs=[x, y], outputs=fc, updates=[
    (theta1, grad_desc(fc, theta1)),
    (theta2, grad_desc(fc, theta2))])
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]]).reshape(4, 2)  # training data X
exp_y = np.array([1, 1, 0, 0])  # training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k])  # call out Theano-compiled cost function, it wil auto update weights
    if i % 500 == 0:
        print('Cost: %s' % (cur_cost,))

print("Training done! Let's test it out:")
print(run_forward([0, 1]))
print(run_forward([1, 1]))
print(run_forward([1, 0]))
print(run_forward([0, 0]))

