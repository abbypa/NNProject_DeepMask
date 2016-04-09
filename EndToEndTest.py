import datetime
import os
from FullNetGenerator import *
from ImageUtils import *
from keras.optimizers import SGD


def print_debug(str):
    print '%s: %s' % (datetime.datetime.now(), str)


def binarize_img(data_array, threshold):
    binary_img = np.copy(data_array)
    # all below threshold -> 0, all above -> 1
    binary_img[data_array >= threshold] = 255  # todo
    binary_img[data_array < threshold] = 0
    return binary_img

graph_arch_path = 'Resources/graph_architecture.json'
graph_weights_path = 'Resources/graph_weights.h5'

if os.path.isfile(graph_arch_path) and os.path.isfile(graph_weights_path):
    print_debug('loading net...')
    graph = model_from_json(open(graph_arch_path).read())
    graph.load_weights(graph_weights_path)
else:
    print_debug('creating net...')
    net_generator = FullNetGenerator('Resources/vgg16_graph_weights.h5')
    graph = net_generator.create_full_net()
    print_debug('compiling net...')
    sgd = SGD(lr=0.001, decay=0.00005, momentum=0.9, nesterov=True)
    graph.compile(optimizer=sgd, loss={'seg_output': 'mse', 'score_output': 'mse'})
    print_debug('saving net...')
    json_string = graph.to_json()
    open(graph_arch_path, 'w').write(json_string)
    graph.save_weights(graph_weights_path)

print_debug('reading image...')
img_path = 'Results/423362-1918790-im.png'
im = prepare_local_image(img_path)

print_debug('running net...')
predictions = graph.predict({'input': im})
binary_mask = binarize_img(predictions['seg_output'][0], 0.1)
save_array_as_img(binary_mask, 'Predictions/0.png')
print predictions['score_output'][0]

expected_mask_path = 'Results/423362-1918790-mask.png'
expected_mask = prepare_expected_mask(expected_mask_path)
expected_result = 1
expected_result_arr = np.array([expected_result])
expected_result_arr = np.expand_dims(expected_result_arr, axis=0)

epochs = 10
for i in range(epochs):
    print_debug('round %d:' % (i+1))
    history = graph.fit({'input': im, 'seg_output': expected_mask, 'score_output': expected_result_arr}, nb_epoch=1)
    predictions = graph.predict({'input': im})
    binary_mask = binarize_img(predictions['seg_output'][0], 0.1)
    prediction_path = 'Predictions/%d.png' % (i+1)
    save_array_as_img(binary_mask, prediction_path)
    print predictions['score_output'][0]


print_debug('done!')

# history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=10)

"""
model.compile(loss='categorical_crossentropy', optimizer=rms)

model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


graph.compile(optimizer='rmsprop', loss={'output1':'mse', 'output2':'mse'})
history = graph.fit({'input':X_train, 'output1':y_train, 'output2':y2_train}, nb_epoch=10)
"""

