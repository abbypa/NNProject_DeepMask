import CreatedVggANetProvider;

# init
vggNetProvider = CreatedVggANetProvider()

# build net
vggNet = vggNetProvider.get_vgg_net()
# transform to a graph
# append segmentation branch
# append classification branch
