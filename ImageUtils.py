import cv2
import urllib
import numpy as np
from PIL import Image
from Constants import mask_pic_true_color, mask_pic_false_color, input_pic_size, output_mask_size


def prepare_local_images(img_paths, normalize=True, resize=False):
    prepared_imgs = [prepare_image(cv2.imread(img_path), normalize=normalize, resize=resize) for img_path in img_paths]
    return np.stack(prepared_imgs)


def prepare_url_image(img_url):
    resp = urllib.urlopen(img_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return prepare_image(image)


def prepare_image(img, normalize=True, resize=False):
    if img.shape != (input_pic_size, input_pic_size, 3):
        print 'Wrong img size!'
        if resize:
            im = cv2.resize(img, (input_pic_size, input_pic_size)).astype(np.float32)
        else:
            return None
    else:
        im = img.astype(np.float32)
    # normalization according to mean pixel value (provided by vgg training)
    if normalize:
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
    # change dimension order - 224x224x3 -> 3x224x224
    im = im.transpose((2, 0, 1))
    return im


def prepare_expected_masks(mask_paths):
    prepared_masks = [prepare_expected_mask(mask_path) for mask_path in mask_paths]
    return np.stack(prepared_masks)


def prepare_expected_mask(mask_path):
    im = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (output_mask_size, output_mask_size)).astype(np.float32)
    # replace visible color with 1 (actual mask)
    im[im > 0] = 1
    # 0 -> -1
    im[im == 0] = -1
    return im


def save_array_as_img(data_array, output_path):
    im = Image.fromarray(data_array)
    im = im.convert("L")
    im.save(output_path)


def binarize_img(data_array, threshold):
    binary_img = np.copy(data_array)
    # all below threshold -> 0, all above -> 1
    binary_img[data_array >= threshold] = mask_pic_true_color
    binary_img[data_array < threshold] = mask_pic_false_color
    return binary_img


def binarize_and_save_mask(data_array, threshold, output_path):
    binary_mask = binarize_img(data_array, threshold)
    save_array_as_img(binary_mask, output_path)