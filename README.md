# NNProject - DeepMask

This is a Keras-based Python implementation of DeepMask- a complex deep neural network for learning object segmentation masks. The full article can be found here: [Learning to Segment Object Candidates](http://arxiv.org/abs/1506.06204).

This was implemented as a final project for TAU Deep Learning course (2016).

### Required installations
This was run on Windows 8.1.

- Python
  - [Anaconda for Windows x64 with Python 2.7](https://www.continuum.io/downloads)
- Theano
  - conda install mingw libpython
  - git clone https://github.com/Theano/Theano.git
  - python setup.py install
- Keras
  - conda install keras
- Open CV
  - Download installation [here](http://opencv.org/)
  - Copy cv2.pyd to site-packages dir inside python's lib dir in anaconda
- Coco API
  - git clone https://github.com/pdollar/coco
  - python setup.py build_ext install

### Required downloads
- VGG-D
  - Net was taken from [here](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3). Weights should be downloaded locally, and converted to graph format using **HelperScripts/CreateVggGraphWeights.py**
  - Full article: [Very Deep Convolutional Networks for Large-Scale Image Recognition](http://arxiv.org/abs/1409.1556)
- MSCOCO
  - Download annotation files [here](http://mscoco.org/dataset/#download)

