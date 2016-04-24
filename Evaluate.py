import datetime
import glob
import os
from Constants import mask_threshold
from FullNetGenerator import *
from ImageUtils import *
from keras.optimizers import SGD
from Losses import *

sgd_lr = 0.001
sgd_decay = 0.00005
sgd_momentum = 0.9

epochs = 1
batch_size = 32
evaluation_batch_size = batch_size

first_round = 1

# paths:
graph_arch_path = 'Resources/graph_architecture_with_transfer.json'
graph_weights_path = 'Predictions/nets/latest_net'
train_predictions_path = 'Predictions/train_predictions'
test_predictions_path = 'Predictions/test_predictions'
train_images_path = 'Predictions/train'
test_images_path = 'Predictions/test'


def print_debug(str_to_print):
    print '%s: %s' % (datetime.datetime.now(), str_to_print)


def evaluate_net_loss(net, train_images, train_expected_scores, train_expected_masks,
                      test_images, test_expected_scores, test_expected_masks):
    train_loss = net.evaluate({'input': train_images, 'score_output': train_expected_scores,
                               'seg_output': train_expected_masks}, batch_size=evaluation_batch_size, verbose=0)
    test_loss = net.evaluate({'input': test_images, 'score_output': test_expected_scores,
                              'seg_output': test_expected_masks}, batch_size=evaluation_batch_size, verbose=0)
    print_debug('evaluation- train loss %s test loss %s' % (train_loss, test_loss))
    return train_loss, test_loss


def evaluate_net_predictions_if_needed(net, round_num, train_images, test_images,
                                       train_expected_scores, test_expected_scores):
    print_debug('evaluating train predictions:')
    evaluate_net_predictions(net, round_num, train_images, train_predictions_path,
                             train_expected_scores)
    print_debug('evaluating test predictions:')
    evaluate_net_predictions(net, round_num, test_images, test_predictions_path,
                             test_expected_scores)


def evaluate_net_predictions(net, round_num, images, predictions_path, expected_scores):
    predictions = net.predict({'input': images})
    score_predictions = predictions['score_output']
    # correct predictions have the same sign
    correct_predictions = sum(map(lambda net_score, expected_score: np.sign(net_score) == np.sign(expected_score),
                                  score_predictions, expected_scores))
    print_debug('%d/%d correct prediction' % (correct_predictions, len(score_predictions)))

    for i in range(len(predictions['seg_output'])):
        mask = predictions['seg_output'][i]
        prediction_path = '%s/round%d-pic%d.png' % (predictions_path, round_num, i)
        binarize_and_save_mask(mask, mask_threshold, prediction_path)

    return correct_predictions


def load_saved_net():
    print_debug('loading net...')
    net = model_from_json(open(graph_arch_path).read())
    net.load_weights(graph_weights_path)
    return net


def compile_net(net):
    print_debug('compiling net...')
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_momentum, nesterov=True)
    net.compile(optimizer=sgd, loss={'score_output': binary_regression_error,
                                     'seg_output': mask_binary_regression_error})
    return net


def example_name_to_result(ex_name):
    if ex_name.startswith('pos'):
        return 1
    elif ex_name.startswith('neg'):
        return -1
    else:
        print 'illegal example: %s' % ex_name
        raise Exception


def prepare_data(examples_path):
    ex_paths = glob.glob('%s/*-im.png' % examples_path)
    # np.random.shuffle(ex_paths)
    images = prepare_local_images(ex_paths)

    expected_mask_paths = [str.replace(img_path, 'im', 'mask') for img_path in ex_paths]
    expected_masks = prepare_expected_masks(expected_mask_paths)

    expected_results = [example_name_to_result(os.path.basename(ex_path)) for ex_path in ex_paths]
    expected_result_arr = np.array([[res] for res in expected_results])

    return images, expected_result_arr, expected_masks


def main():
    graph = load_saved_net()
    compile_net(graph)  # current keras version cannot load compiled net with custom loss function
    print_debug('preparing data...')

    train_images, train_expected_scores, train_expected_masks = prepare_data(train_images_path)
    test_images, test_expected_scores, test_expected_masks = prepare_data(test_images_path)
    print_debug('Dataset- %d train examples, %d test examples' %
                (len(train_expected_scores), len(test_expected_scores)))

    print_debug('Evaluating...')
    evaluate_net_loss(graph, train_images, train_expected_scores, train_expected_masks,
                      test_images, test_expected_scores, test_expected_masks)
    evaluate_net_predictions_if_needed(graph, first_round, train_images, test_images,
                                       train_expected_scores, test_expected_scores)


if __name__ == "__main__":
    main()
