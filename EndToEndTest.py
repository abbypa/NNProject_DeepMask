import datetime
import os
from FullNetGenerator import *
from ImageUtils import *
from keras.optimizers import SGD
from Losses import *

graph_arch_path = 'Resources/graph_architecture.json'
graph_weights_path = 'Resources/graph_weights.h5'
original_net_weights_path = None  # TODO- 'Resources/vgg16_graph_weights.h5'


def print_debug(str):
    print '%s: %s' % (datetime.datetime.now(), str)


def binarize_img(data_array, threshold):
    binary_img = np.copy(data_array)
    # all below threshold -> 0, all above -> 1
    binary_img[data_array >= threshold] = 255  # todo
    binary_img[data_array < threshold] = 0
    return binary_img


def test_prediction(imgs, round_num, net, expected_result_arr, expected_masks):
    predictions = net.predict({'input': imgs})
    binary_mask = binarize_img(predictions['seg_output'][0], 0.1)
    prediction_path = 'Predictions/%d.png' % (round_num)
    save_array_as_img(binary_mask, prediction_path)
    print_debug('prediction %s' % predictions['score_output'])
    # evaluation = net.evaluate({'input': imgs, 'score_output': expected_result_arr, 'seg_output': expected_masks},
                              # batch_size=1)
    # print_debug('evaluation %s' % evaluation)


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
    net = net_generator.create_full_net()
    print_debug('net created:')
    print net.summary()
    return net


def compile_net(net):
    print_debug('compiling net...')
    # sgd = SGD(lr=0.001, decay=0.00005, momentum=0.9, nesterov=True)
    sgd = SGD(lr=0.0001)
    # TODO- loss_weights
    net.compile(optimizer=sgd, loss={'score_output': binary_regression_error, 'seg_output': mask_binary_regression_error})  # TODO
    return net


def save_net(net):
    print_debug('saving net...')
    json_string = net.to_json()
    open(graph_arch_path, 'w').write(json_string)
    net.save_weights(graph_weights_path)


def prepare_data():
    print_debug('preparing data...')
    img_paths = [
        'Results/423362-1918790-im.png',
        #'Results/49-254537-im.png',
        #'Results/49-1211660-im.png',
        #'Results/49-2010752-im.png',
        #'Results/61-434050-im.png',
        #'Results/61-555226-im.png',
        #'Results/61-580815-im.png',
        #'Results/61-2012396-im.png',
        #'Results/71-141916-im.png',
        #'Results/71-147406-im.png',
        ]
    images = prepare_local_images(img_paths)

    expected_mask_paths = [str.replace(img_path, 'im', 'mask') for img_path in img_paths]
    expected_masks = prepare_expected_masks(expected_mask_paths)

    expected_result = 1
    expected_result_arr = np.array([expected_result])
    expected_result_arr = np.tile(expected_result_arr, (len(img_paths), 1))

    return [images, expected_result_arr, expected_masks]


def main():
    if False and saved_net_exists():  # TODO
        graph = load_saved_net()
    else:
        graph = create_net()
        compile_net(graph)
        # save_net(graph) TODO

    [images, expected_result_arr, expected_masks] = prepare_data()

    print_debug('running net...')
    test_prediction(images, 0, graph, expected_result_arr, expected_masks)

    epochs = 10
    for i in range(epochs):
        print_debug('starting round %d:' % (i+1))
        history = graph.fit({'input': images, 'seg_output': expected_masks, 'score_output': expected_result_arr},
                            nb_epoch=1, verbose=0)
        print_debug('fit loss: %s' % history.history['loss'])
        test_prediction(images, i+1, graph, expected_result_arr, expected_masks)


if __name__ == "__main__":
    main()
