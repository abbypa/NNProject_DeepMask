import cv2
import urllib
import numpy as np


def prepare_local_image(img_path):
    image = cv2.imread(img_path)
    return prepare_image(image)


def prepare_url_image(img_url):
    resp = urllib.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return prepare_image(image)


def prepare_image(img):
    im = cv2.resize(img, (224, 224)).astype(np.float32)
    im[:, :, 0] -= 103.939
    im[:, :, 1] -= 116.779
    im[:, :, 2] -= 123.68
    im = im.transpose((2, 0, 1))
    im = np.expand_dims(im, axis=0)
    return im
