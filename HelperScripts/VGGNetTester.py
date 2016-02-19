from ReadyVggDNetProvider import *
from ImageUtils import *
from keras.optimizers import SGD


def test_img_graph(graph, img_path):
    im = prepare_local_image(img_path)
    out = graph.predict({'input': im})
    return np.argmax(out['output'])


def test_partial_net():
    graph = load_and_alter_net('..\Resources\\vgg16_graph_weights.h5')
    print graph.summary()
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'newoutput': 'categorical_crossentropy'})
    im = prepare_local_image('..\Resources\\img-cat2.jpg')
    out = graph.predict({'input': im})
    print 'testing...'
    print out

if __name__ == "__main__":
    print 'creating graph model...'
    graph = VGG_16_graph('..\Resources\\vgg16_graph_weights.h5')
    print graph.summary()

    print 'compiling graph...'
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'output': 'categorical_crossentropy'})

    print 'testing...'
    print test_img_graph(graph, '..\Resources\\cat2.jpg')
    print test_img_graph(graph, '..\Resources\\cat.jpg')
    print test_img_graph(graph, '..\Resources\\img-cat2.jpg')
    print test_img_graph(graph, '..\Resources\\img-cat.jpg')
    print test_img_graph(graph, '..\Resources\\img-zebra.jpg')
    print test_img_graph(graph, '..\Resources\\img-zebra2.jpg')

    print 'creating partial net...'
    test_partial_net()
