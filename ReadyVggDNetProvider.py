from keras.models import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np


def VGG_16_graph(weights_path=None, with_output=True):
    model = Graph()
    model.add_input(name='input', input_shape=(3, 224, 224))
    model.add_node(ZeroPadding2D((1,1)), name='pad1', input='input')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='conv1', input='pad1')
    model.add_node(ZeroPadding2D((1,1)), name='pad2', input='conv1')
    model.add_node(Convolution2D(64, 3, 3, activation='relu'), name='conv2', input='pad2')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool1', input='conv2')

    model.add_node(ZeroPadding2D((1,1)), name='pad3', input='pool1')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='conv3', input='pad3')
    model.add_node(ZeroPadding2D((1,1)), name='pad4', input='conv3')
    model.add_node(Convolution2D(128, 3, 3, activation='relu'), name='conv4', input='pad4')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool2', input='conv4')

    model.add_node(ZeroPadding2D((1,1)), name='pad5', input='pool2')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='conv5', input='pad5')
    model.add_node(ZeroPadding2D((1,1)), name='pad6', input='conv5')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='conv6', input='pad6')
    model.add_node(ZeroPadding2D((1,1)), name='pad7', input='conv6')
    model.add_node(Convolution2D(256, 3, 3, activation='relu'), name='conv7', input='pad7')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool3', input='conv7')

    model.add_node(ZeroPadding2D((1,1)), name='pad8', input='pool3')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv8', input='pad8')
    model.add_node(ZeroPadding2D((1,1)), name='pad9', input='conv8')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv9', input='pad9')
    model.add_node(ZeroPadding2D((1,1)), name='pad10', input='conv9')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv10', input='pad10')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool4', input='conv10')

    model.add_node(ZeroPadding2D((1,1)), name='pad11', input='pool4')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv11', input='pad11')
    model.add_node(ZeroPadding2D((1,1)), name='pad12', input='conv11')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv12', input='pad12')
    model.add_node(ZeroPadding2D((1,1)), name='pad13', input='conv12')
    model.add_node(Convolution2D(512, 3, 3, activation='relu'), name='conv13', input='pad13')
    model.add_node(MaxPooling2D((2,2), strides=(2,2)), name='pool5', input='conv13')

    model.add_node(Flatten(), name='flat', input='pool5')
    model.add_node(Dense(4096, activation='relu'), name='dense1', input='flat')
    model.add_node(Dropout(0.5), name='drop1', input='dense1')
    model.add_node(Dense(4096, activation='relu'), name='dense2', input='drop1')
    model.add_node(Dropout(0.5), name='drop2', input='dense2')
    model.add_node(Dense(1000, activation='softmax'), name='dense3', input='drop2')

    if with_output:
        model.add_output(input='dense3', name='output')

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_and_alter_net(with_output=True):
    model = VGG_16_graph('Resources\\vgg16_graph_weights.h5', False)
    nodes_to_pop = ['dense3', 'drop2', 'dense2', 'drop1', 'dense1', 'flat', 'pool5']
    params_to_pop = 2 * 3  # 2 params for each dense layer

    # remove old output
    if len(model.outputs) > 0:
        model.outputs.pop('output')
        model.output_order.pop()
        model.output_config.pop()
    # remove redundant layers (from the end)
    for node in nodes_to_pop:
        model.nodes.pop(node)
    # remove relevant params
    for _ in range(params_to_pop):
        model.params.pop()
    # add a new output
    if with_output:
        model.add_output(name='newoutput', input='conv13')
    return model


def prepare_img(img_path):
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


def test_img_graph(graph, img_path):
    im = prepare_img(img_path)
    out = graph.predict({'input': im})
    return np.argmax(out['output'])


def test_partial_net():
    graph = load_and_alter_net()
    print graph.summary()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'newoutput': 'categorical_crossentropy'})
    im = prepare_img('Resources\\img-cat2.jpg')
    out = graph.predict({'input': im})
    print out

if __name__ == "__main__":
    test_partial_net()

    print 'creating graph model...'
    graph = VGG_16_graph('Resources\\vgg16_graph_weights.h5')

    print 'compiling graph...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    print test_img_graph(graph, 'Resources\\cat2.jpg')
    print test_img_graph(graph, 'Resources\\cat.jpg')
    print test_img_graph(graph, 'Resources\\img-cat2.jpg')
    print test_img_graph(graph, 'Resources\\img-cat.jpg')
    print test_img_graph(graph, 'Resources\\img-zebra.jpg')
    print test_img_graph(graph, 'Resources\\img-zebra2.jpg')

    print graph.summary()


    # todo- delete testing related functions and/or move to a separate file
    # todo- delete all prints
