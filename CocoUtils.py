from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
from PIL import Image, ImageDraw


class CocoUtils(object):
    def __init__(self, data_dir, data_type):
        ann_file = '%s/annotations/instances_%s.json' % (data_dir, data_type)
        # initialize COCO api for instance annotations
        self.coco = COCO(ann_file)

    def get_img_annotations(self, pic_id):
        ann_ids = self.coco.getAnnIds(imgIds=pic_id, iscrowd=None)
        return self.coco.loadAnns(ann_ids)

    def get_mask_array_and_image(self, annotation, img_width, img_height, fill_color):
        seg = annotation['segmentation']
        raster_img = Image.new('L', (img_width, img_height), 0)
        for polyg in seg:
            ImageDraw.Draw(raster_img).polygon(polyg, outline=fill_color, fill=fill_color)
        return np.array(raster_img), raster_img

    def get_annotation_mask(self, annotation, img_width, img_height):
        seg_mask, seg_img = self.get_mask_array_and_image(annotation, img_width, img_height, 1)
        return seg_mask

    def get_annotation_image(self, annotation, img_width, img_height):
        seg_mask, seg_img = self.get_mask_array_and_image(annotation, img_width, img_height, 128)
        return seg_img

    def show_annotations(self, pic_path, annotations):
        pylab.rcParams['figure.figsize'] = (10.0, 8.0)
        read_img = io.imread(pic_path)
        plt.figure()
        plt.imshow(read_img)
        self.coco.showAnns(annotations)

    def get_and_show_annotations(self, pic_path, pic_id):
        self.show_annotations(pic_path, self.get_img_annotations(pic_id))

    def is_segmentation_centered(self, segmentation, img_width, img_height):
        bbox = segmentation['bbox']
        bbox_center_x = (bbox[2] + bbox[0]) / 2
        bbox_center_y = (bbox[3] + bbox[1]) / 2
        # todo- what is 'centered'? what size?
        pic_center_x = img_width/2
        pic_center_y = img_height/2
        return abs(pic_center_x-bbox_center_x) <= 65 and abs(pic_center_y-bbox_center_y) <= 65


coco_utils = CocoUtils('..', 'val2014')
picId = 438915
img_path = '../images/%s.jpg' % picId
img = coco_utils.coco.loadImgs(picId)[0]
coco_utils.get_and_show_annotations(img_path, picId)
anns = coco_utils.get_img_annotations(picId)

# for image showing, should replace 1's with 128, otherwise it's pretty much invisible
centered_anns = []
for ann_num in range(len(anns)):
    im = coco_utils.get_annotation_image(anns[ann_num], img['width'], img['height'])
    im.save('Results/' + str(ann_num) + '.jpg', 'JPEG')
    if coco_utils.is_segmentation_centered(anns[ann_num], img['width'], img['height']):
        print '%s is centered' % ann_num
        centered_anns.append(anns[ann_num])
coco_utils.show_annotations(img_path, centered_anns)
print 'done'

# these lines causes errors with ntdll. might also happen in showAnns, like in image 262148
# as an alternative, I use ImageDraw to manually calculate the mask

# Rs = frPyObjects(ann['segmentation'], img['height'], img['width'])
# masks = decode(Rs)

# todo- from web
# resp = urllib.urlopen(img_url)
#    image = np.asarray(bytearray(resp.read()), dtype="uint8")
#    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
