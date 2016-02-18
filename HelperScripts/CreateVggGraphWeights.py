from keras.models import *
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import urllib


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


def VGG_16_graph():
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
    return model


def prepare_local_image(img_path):
    image = cv2.imread(img_path)
    return prepare_img(image)


def prepare_url_image(img_url):
    resp = urllib.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return prepare_img(image)


def prepare_img(img):
    im = cv2.resize(img, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


def test_img(model, im):
    out = model.predict(im)
    return np.argmax(out)


def test_img_graph(graph, im):
    out = graph.predict({'input': im})
    return np.argmax(out['output'])


if __name__ == "__main__":
    print 'creating graph model...'
    graph = VGG_16_graph()

    print 'creating sequential model...'
    model = VGG_16('..\\Resources\\vgg16_weights.h5')

    print 'setting graph weights...'
    graph.set_weights(model.get_weights())
    graph.save_weights('vgg16_graph_weights.h5')

    print 'compiling graph model...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    print 'compiling sequential model...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    print 'TESTING...'
    urls = [
        "http://farm3.static.flickr.com/2793/4253075042_158b41a887.jpg",
        "http://farm4.static.flickr.com/3279/2405903773_4e3573e73e.jpg",
        "http://farm1.static.flickr.com/79/275508357_b0ac39adbb.jpg",
        "http://farm1.static.flickr.com/250/456602357_a15b197caa.jpg"
    ]

    for url in urls:
        im = prepare_url_image(url)
        print 'graph result: ', test_img_graph(graph, im)
        print 'sequence result: ', test_img(model, im)
