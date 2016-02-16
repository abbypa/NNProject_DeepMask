import CreatedVggANetProvider
import ExamplesCreator

# config
vgg_net_provider = CreatedVggANetProvider()
examples_creator = ExamplesCreator()

# create examples
positive_examples = examples_creator.create_positive_examples()
negative_examples = examples_creator.create_negative_examples()

# build net
vggNet = vgg_net_provider.get_vgg_net()
# transform to a graph
# append segmentation branch
# append classification branch
# train
