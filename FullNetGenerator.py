from VggDNetGraphProvider import *
from keras.layers.core import Reshape
# TODO from keras.layers.convolutional import UpSampling2D


class FullNetGenerator(object):
    def __init__(self, weights_path):
        self.final_common_layer = 'conv13'
        self.weights_path = weights_path

    def create_full_net(self, score_branch=True, seg_branch=True):
        vgg_provider = VggDNetGraphProvider()
        net = vgg_provider.get_vgg_partial_graph(weights_path=self.weights_path, with_output=False)
        if score_branch:
            self.append_score_branch(net)
        if seg_branch:
            self.append_segmentation_branch(net)
        return net

    def append_score_branch(self, graph):
        graph.add_node(MaxPooling2D((2, 2), strides=(2, 2)), name='score_pool1', input=self.final_common_layer)
        graph.add_node(Flatten(), name='score_flat', input='score_pool1')
        graph.add_node(Dense(512, activation='relu'), name='score_dense1', input='score_flat')
        graph.add_node(Dropout(0.5), name='score_drop1', input='score_dense1')
        graph.add_node(Dense(1024, activation='relu'), name='score_dense2', input='score_drop1')
        graph.add_node(Dropout(0.5), name='score_drop2', input='score_dense2')
        graph.add_node(Dense(1), name='score_linear', input='score_drop2')
        graph.add_output(input='score_linear', name='score_output')

    def append_segmentation_branch(self, graph):
        graph.add_node(Convolution2D(512, 1, 1, activation='relu'), name='seg_conv1', input=self.final_common_layer)
        graph.add_node(Flatten(), name='seg_flat', input='seg_conv1')
        graph.add_node(Dense(512), name='seg_dense1', input='seg_flat')  # no activation here!
        graph.add_node(Dense(56*56), name='seg_dense2', input='seg_dense1')
        graph.add_node(Reshape(dims=(56, 56)), name='seg_reshape', input='seg_dense2')
        # graph.add_node(UpSampling2D(size=(4, 4)), name='seg_upsample', input='seg_reshape')
        # TODO- bilinear upsampling layer
        graph.add_output(input='seg_reshape', name='seg_output')


# usage-
# fng = FullNetGenerator('Resources/vgg16_graph_weights.h5')
# fn = fng.create_full_net()
