import datetime
import os
from FullNetGenerator import *
from ImageUtils import *
from keras.optimizers import SGD

graph_arch_path = 'Resources/graph_architecture_score.json'
graph_weights_path = 'Resources/graph_weights_score.h5'
original_net_weights_path = None  # TODO- 'Resources/vgg16_graph_weights.h5'


def print_debug(str):
    print '%s: %s' % (datetime.datetime.now(), str)


def binarize_img(data_array, threshold):
    binary_img = np.copy(data_array)
    # all below threshold -> 0, all above -> 1
    binary_img[data_array >= threshold] = 255  # todo
    binary_img[data_array < threshold] = 0
    return binary_img


def test_prediction(im, round_num, net):
    predictions = net.predict({'input': im})
    # binary_mask = binarize_img(predictions['seg_output'][0], 0.1)
    # prediction_path = 'Predictions/%d.png' % (round_num)
    # save_array_as_img(binary_mask, prediction_path)
    print_debug('prediction %s' % predictions['score_output'][0])


def saved_net_exists():
    return os.path.isfile(graph_arch_path) and os.path.isfile(graph_weights_path)


def load_saved_net():
    print_debug('loading net...')
    net = model_from_json(open(graph_arch_path).read())
    net.load_weights(graph_weights_path)
    return net


def create_net():
    print_debug('creating net...')
    net_generator = FullNetGenerator(original_net_weights_path)
    net = net_generator.create_full_net(seg_branch=False) # TODO
    print_debug('net created:')
    print net.summary()
    return net


def compile_net(net):
    print_debug('compiling net...')
    # sgd = SGD(lr=0.001, decay=0.00005, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.00001)
    # TODO- loss_weights
    # graph.compile(optimizer=sgd, loss={'seg_output': 'mse', 'score_output': 'mse'})
    net.compile(optimizer=sgd, loss={'score_output': 'mae'})
    return net


def save_net(net):
    print_debug('saving net...')
    json_string = net.to_json()
    open(graph_arch_path, 'w').write(json_string)
    net.save_weights(graph_weights_path)

if False and saved_net_exists: # TODO
    graph = load_saved_net()
else:
    graph = create_net()
    compile_net(graph)
    # save_net(graph)

print_debug('reading image...')
img_path = 'Results/423362-1918790-im.png'
im = prepare_local_image(img_path)

print_debug('running net...')
# test_prediction(im, 0)

expected_mask_path = 'Results/423362-1918790-mask.png'
expected_mask = prepare_expected_mask(expected_mask_path)
expected_result = 1
expected_result_arr = np.array([expected_result])
expected_result_arr = np.expand_dims(expected_result_arr, axis=0)

epochs = 10
for i in range(epochs):
    print_debug('round %d:' % (i+1))
    # history = graph.fit({'input': im, 'seg_output': expected_mask, 'score_output': expected_result_arr}, nb_epoch=1, verbose=0)
    history = graph.fit({'input': im, 'score_output': expected_result_arr}, nb_epoch=1, verbose=0)
    # test_prediction(im, i+1)
    print_debug('fit loss: %s' % history.history['loss'])
    img_path = 'Results/202617-256996-im.png'
    im = prepare_local_image(img_path)
    # todo- each batch with multiple imgs


print_debug('done!')
