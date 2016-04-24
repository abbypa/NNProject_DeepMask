import datetime
import glob
import os
from Constants import mask_threshold
from FullNetGenerator import *
from ImageUtils import *
from keras.optimizers import SGD
from Losses import *
import math

sgd_lr = 0.001
sgd_decay = 0.00005
sgd_momentum = 0.9

epochs = 1
batch_size = 32
evaluation_batch_size = batch_size
critical_loss = 500

first_round = 1

rounds = 1
rounds_to_backup_weights = 5
rounds_to_predict_results = 2


# paths:
graph_arch_path = 'Resources/graph_architecture_with_transfer.json'
graph_weights_path = 'Resources/graph_weights_with_transfer.h5'
original_net_weights_path = 'Resources/vgg16_graph_weights.h5'
train_predictions_path = 'Predictions/train_predictions'
test_predictions_path = 'Predictions/test_predictions'
nets_dir_path = 'Predictions/nets'
loss_file_path = 'Predictions/out-loss.csv'
score_predictions_file_path = 'Predictions/score_predictions.csv'
train_images_path = 'Predictions/train'
test_images_path = 'Predictions/test'


def print_debug(str_to_print):
    print '%s: %s' % (datetime.datetime.now(), str_to_print)


def evaluate_net_loss(net, train_images, train_expected_scores, train_expected_masks,
                      test_images, test_expected_scores, test_expected_masks, loss_file):
    train_loss = net.evaluate({'input': train_images, 'score_output': train_expected_scores,
                               'seg_output': train_expected_masks}, batch_size=evaluation_batch_size, verbose=0)
    test_loss = net.evaluate({'input': test_images, 'score_output': test_expected_scores,
                              'seg_output': test_expected_masks}, batch_size=evaluation_batch_size, verbose=0)
    print_debug('evaluation- train loss %s test loss %s' % (train_loss, test_loss))
    loss_file.write('%s,%s,%s\n' % (datetime.datetime.now(), train_loss, test_loss))
    loss_file.flush()
    return train_loss, test_loss


def evaluate_net_predictions_if_needed(net, round_num, train_images, test_images,
                                       train_expected_scores, test_expected_scores, score_predictions_file):
    if round_num % rounds_to_predict_results == 0:
        print_debug('evaluating train predictions:')
        correct_scores_train = evaluate_net_predictions(net, round_num, train_images, train_predictions_path,
                                                        train_expected_scores)
        print_debug('evaluating test predictions:')
        correct_scores_test = evaluate_net_predictions(net, round_num, test_images, test_predictions_path,
                                                       test_expected_scores)
        score_predictions_file.write('%d,%d,%d,%d\n' % (correct_scores_train, len(train_expected_scores),
                                                        correct_scores_test, len(test_expected_scores)))
        score_predictions_file.flush()


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
    sgd = SGD(lr=sgd_lr, decay=sgd_decay, momentum=sgd_momentum, nesterov=True)
    net.compile(optimizer=sgd, loss={'score_output': binary_regression_error,
                                     'seg_output': mask_binary_regression_error})
    return net


def save_net(net):
    print_debug('saving net...')
    json_string = net.to_json()
    open(graph_arch_path, 'w').write(json_string)
    net.save_weights(graph_weights_path)


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
    # np.random.shuffle(ex_paths) # shuffle wil be set later, for consistency in loss calculation
    images = prepare_local_images(ex_paths)

    expected_mask_paths = [str.replace(img_path, 'im', 'mask') for img_path in ex_paths]
    expected_masks = prepare_expected_masks(expected_mask_paths)

    expected_results = [example_name_to_result(os.path.basename(ex_path)) for ex_path in ex_paths]
    expected_result_arr = np.array([[res] for res in expected_results])

    return images, expected_result_arr, expected_masks


def backup_net(graph, round_num):
    print_debug("Saving latest net weights")
    graph.save_weights('%s/latest_net' % nets_dir_path, overwrite=True)

    print_debug("Saving net weights for round %d" % round_num)
    if round_num % rounds_to_backup_weights == 0:
        graph.save_weights('%s/net-round%d' % (nets_dir_path, round_num))


def is_exploding_loss(loss):
    math.isnan(loss) or math.isinf(loss) or loss >= critical_loss


def main():
    train_losses = []
    test_losses = []

    loss_file = open(loss_file_path, 'a')
    loss_file.write('time,train loss,test loss\n')
    score_predictions_file = open(score_predictions_file_path, 'a')
    score_predictions_file.write('train success, train total, test success, test total\n')

    if saved_net_exists():
        graph = load_saved_net()
    else:
        graph = create_net()
        save_net(graph)

    compile_net(graph)  # current keras version cannot load compiled net with custom loss function
    print_debug('preparing data...')

    train_images, train_expected_scores, train_expected_masks = prepare_data(train_images_path)
    test_images, test_expected_scores, test_expected_masks = prepare_data(test_images_path)
    print_debug('Dataset- %d train examples, %d test examples' %
                (len(train_expected_scores), len(test_expected_scores)))

    # uncomment to restore last run results / initial net result
    # print_debug('running net...')
    # losses.append(test_prediction(train_images, last_i, graph, train_expected_scores, train_expected_masks, out))

    for round_number in range(first_round, rounds + 1):
        print_debug('starting round %d:' % round_number)
        graph.fit({'input': train_images, 'seg_output': train_expected_masks, 'score_output': train_expected_scores},
                  nb_epoch=epochs, batch_size=batch_size, verbose=0, shuffle=True)
        print_debug('Evaluating...')
        train_loss, test_loss = evaluate_net_loss(graph, train_images, train_expected_scores, train_expected_masks,
                                                  test_images, test_expected_scores, test_expected_masks, loss_file)
        if is_exploding_loss(train_loss):
            print_debug("Loss %s too big- stopping" % train_loss)
            break

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        evaluate_net_predictions_if_needed(graph, round_number, train_images, test_images,
                                           train_expected_scores, test_expected_scores, score_predictions_file)
        backup_net(graph, round_number)

    loss_file.close()


if __name__ == "__main__":
    main()
