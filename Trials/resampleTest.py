import cv2
from PIL import Image
from scipy import ndimage

pic = cv2.imread('Trial Resources/demo.png')
new_pic = ndimage.zoom(pic, (2, 2, 1), order=1)

im = Image.fromarray(new_pic)
im = im.convert("L")
im.save('Trial Resources/res.png')
