from keras.models import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def VGG_16_graph(weights_path=None):
    #sequence_model = VGG_16('Resources\\vgg16_weights.h5')
    #print sequence_model.weights

    model = Graph()
    model.add_input(name='input', input_shape=(3, 224, 224))
    model.add_node(ZeroPadding2D((1,1)), name='pad1', input='input')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='relu1', input='pad1') # weights=sequence_model.layers[1].W.container
    model.add_node(ZeroPadding2D((1,1)), name='pad2', input='relu1')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='relu2', input='pad2')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool1', input='relu2')

    model.add_node(ZeroPadding2D((1,1)), name='1', input='pool1')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='2', input='1')
    model.add_node(ZeroPadding2D((1,1)), name='3', input='2')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='4', input='3')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='5', input='4')

    model.add_node(ZeroPadding2D((1,1)), name='6', input='5')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='7', input='6')
    model.add_node(ZeroPadding2D((1,1)), name='8', input='7')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='9', input='8')
    model.add_node(ZeroPadding2D((1,1)), name='10', input='9')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='11', input='10')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='12', input='11')

    model.add_node(ZeroPadding2D((1,1)), name='13', input='12')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='14', input='13')
    model.add_node(ZeroPadding2D((1,1)), name='15', input='14')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='16', input='15')
    model.add_node(ZeroPadding2D((1,1)), name='17', input='16')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='18', input='17')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='19', input='18')

    model.add_node(ZeroPadding2D((1,1)), name='20', input='19')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='21', input='20')
    model.add_node(ZeroPadding2D((1,1)), name='22', input='21')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='23', input='22')
    model.add_node(ZeroPadding2D((1,1)), name='24', input='23')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='25', input='24')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='26', input='25')

    model.add_node(Flatten(), name='27', input='26')
    model.add_node(Dense(4096, activation='relu'), name='28', input='27')
    model.add_node(Dropout(0.5), name='29', input='28')
    model.add_node(Dense(4096, activation='relu'), name='30', input='29')
    model.add_node(Dropout(0.5), name='31', input='30')
    model.add_node(Dense(1000, activation='softmax'), name='32', input='31')

    model.add_output(input='32', name='output')

    # if weights_path:
    #     model.load_weights(weights_path)

    return model

def load_and_alter_net():
    model = VGG_16('Resources\\vgg16_weights.h5')
    # remove redundant layers (from the end)
    for i in range(3):
        # dense layer
        model.layers.pop()
        model.params.pop()
        model.params.pop()
        # dropout / flatten
        model.layers.pop()
    # last pooling layer
    model.layers.pop()
    return model


def load_net_for_testing():
    model = VGG_16('Resources\\vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def prepare_img(img_path):
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


def test_img(model, img_path):
    im = prepare_img(img_path)
    out = model.predict(im)
    return np.argmax(out)


def test_img_graph(graph, img_path):
    im = prepare_img(img_path)
    out = graph.predict({'input': im})
    return np.argmax(out['output'])


# Test pretrained model
def test_full_net():
    print "loading net..."
    model = load_net_for_testing()
    print "testing..."
    print test_img(model, 'Resources\\cat2.jpg')
    print test_img(model, 'Resources\\cat.jpg')
    print test_img(model, 'Resources\\img-cat2.jpg')
    print test_img(model, 'Resources\\img-cat.jpg')
    print test_img(model, 'Resources\\img-zebra.jpg')
    print test_img(model, 'Resources\\img-zebra2.jpg')


def test_partial_net():
    model = load_and_alter_net()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    im = prepare_img('Resources\\img-cat2.jpg')
    out = model.predict(im)
    print out

if __name__ == "__main__":
    # test_partial_net()
    # test_full_net()
    print 'creating sequential model...'
    graph = VGG_16_graph('Resources\\vgg16_weights.h5')
    # print 'graph get weights'
    # print graph.get_weights()
    print 'creating grapth model...'
    model = VGG_16('Resources\\vgg16_weights.h5')
    # print 'model get weights'
    # print model.get_weights()
    print 'setting graph weights...'
    graph.set_weights(model.get_weights())
    # print 'graph new weights'
    # print graph.get_weights()

    print 'compiling graph...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # graph.compile(optimizer=sgd, loss={'output': 'mse'})
    graph.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    print 'compiling sequential model...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print test_img_graph(graph, 'Resources\\cat2.jpg')
    print test_img_graph(graph, 'Resources\\cat.jpg')
    print test_img_graph(graph, 'Resources\\img-cat2.jpg')
    print test_img_graph(graph, 'Resources\\img-cat.jpg')
    print test_img_graph(graph, 'Resources\\img-zebra.jpg')
    print test_img_graph(graph, 'Resources\\img-zebra2.jpg')
    print test_img(model, 'Resources\\cat2.jpg')
    print test_img(model, 'Resources\\cat.jpg')
    print test_img(model, 'Resources\\img-cat2.jpg')
    print test_img(model, 'Resources\\img-cat.jpg')
    print test_img(model, 'Resources\\img-zebra.jpg')
    print test_img(model, 'Resources\\img-zebra2.jpg')