from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


class CocoUtils(object):
    def __init__(self, data_dir, data_type):
        ann_file = '%s/annotations/instances_%s.json' % (data_dir, data_type)
        # initialize COCO api for instance annotations
        self.coco = COCO(ann_file)

    def get_img_annotations(self, pic_id):
        ann_ids = self.coco.getAnnIds(imgIds=pic_id, iscrowd=None)
        return self.coco.loadAnns(ann_ids)

    def show_annotations(self, pic_path, anns):
        pylab.rcParams['figure.figsize'] = (10.0, 8.0)
        I = io.imread(pic_path)
        plt.figure()
        plt.imshow(I)
        self.coco.showAnns(anns)

    def get_and_show_annotations(self, pic_path, pic_id):
        self.show_annotations(pic_path, self.get_img_annotations(pic_id))

    def is_segmentation_centered(self, segmentation, img):
        bbox = segmentation['bbox']
        bbox_center_x = (bbox[2] + bbox[0]) / 2
        bbox_center_y = (bbox[3] + bbox[1]) / 2
        # todo- what is 'centered'? what size?
        pic_center_x = img['width']/2
        pic_center_y = img['height']/2
        return abs(pic_center_x-bbox_center_x) <= 65 and abs(pic_center_y-bbox_center_y) <= 65


coco_utils = CocoUtils('..', 'val2014')
picId = 438915
img_path = '../images/%s.jpg' % picId
img = coco_utils.coco.loadImgs(picId)[0]
coco_utils.get_and_show_annotations(img_path, picId)
anns = coco_utils.get_img_annotations(picId)
centered_anns = []
for seg in anns:
    if coco_utils.is_segmentation_centered(seg, img):
        centered_anns.append(seg)
coco_utils.show_annotations(img_path, centered_anns)
print 'done'

#todo- from web
# resp = urllib.urlopen(img_url)
#    image = np.asarray(bytearray(resp.read()), dtype="uint8")
#    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
