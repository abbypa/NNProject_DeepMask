from VggDNetGraphProvider import *
from ImageUtils import *
from keras.optimizers import SGD


def test_img_graph(graph, img_path):
    im = prepare_local_images([img_path], resize=True)
    out = graph.predict({'input': im})
    return np.argmax(out['output'])


def test_partial_net():
    graph = netProvider.get_vgg_partial_graph('..\Resources\\vgg16_graph_weights.h5')
    print graph.summary()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'newoutput': 'categorical_crossentropy'})
    im = prepare_local_images(['..\Resources\\old\\img-cat2.jpg'], resize=True)
    out = graph.predict({'input': im})
    print 'testing...'
    print out

if __name__ == "__main__":
    netProvider = VggDNetGraphProvider()

    print 'creating graph model...'
    graph = netProvider.get_vgg_full_graph('..\Resources\\vgg16_graph_weights.h5')
    print graph.summary()

    print 'compiling graph...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    print 'testing...'
    print test_img_graph(graph, '..\Resources\\old\\cat.jpg')
    print test_img_graph(graph, '..\Resources\\old\\cat2.jpg')
    print test_img_graph(graph, '..\Resources\\old\\img-cat2.jpg')
    print test_img_graph(graph, '..\Resources\\old\\img-cat.jpg')
    print test_img_graph(graph, '..\Resources\\old\\img-zebra.jpg')
    print test_img_graph(graph, '..\Resources\\old\\img-zebra2.jpg')

    print 'creating partial net...'
    test_partial_net()
