import cv2
import urllib
import numpy as np
from PIL import Image


def prepare_local_image(img_path):
    image = cv2.imread(img_path)
    return prepare_image(image)


def prepare_url_image(img_url):
    resp = urllib.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return prepare_image(image)


def prepare_image(img, normalize=True):
    im = cv2.resize(img, (224, 224)).astype(np.float32)
    # normalization according to mean pixel value (provided by vgg training)
    if normalize:
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    # change dimension order - 224x224x3 -> 3x224x224
    im = im.transpose((2, 0, 1))
    # add another dimension for bulk processing
    im = np.expand_dims(im, axis=0)
    return im


def prepare_expected_mask(mask_path):
    im = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (56, 56)).astype(np.float32)
    # replace 128 with 1 (visible to actual mask)
    im[:, :] /= 128
    # add another dimension to suit expected format- 56x56 -> 1x56x56
    im = np.expand_dims(im, axis=0)
    return im


def save_array_as_img(data_array, output_path):
    im = Image.fromarray(data_array)
    im = im.convert("L")
    im.save(output_path)
