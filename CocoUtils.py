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

    # mask true's are 1 but image true's are 128- otherwise it's pretty much invisible
    def get_annotation_image(self, annotation, img_width, img_height):
        seg_mask, seg_img = self.get_mask_array_and_image(annotation, img_width, img_height, 128)
        return seg_img

    def are_legal_anotations(self, annotations):
        # unfortunately, only polygon segmentations work for now (RLE mask type decoding causes a python crash)
        polygon_segmentations = ['segmentation' in ann and type(ann['segmentation']) == list for ann in annotations]
        return all(polygon_segmentations)

    def show_annotations(self, pic_path, annotations):
        if self.are_legal_anotations(annotations):
            pylab.rcParams['figure.figsize'] = (10.0, 8.0)
            read_img = io.imread(pic_path)
            plt.figure()
            plt.imshow(read_img)
            self.coco.showAnns(annotations)
        else:
            print 'cannot show invalid annotation'

    def is_segmentation_centered(self, segmentation, img_width, img_height):
        bbox = segmentation['bbox']
        bbox_center_x = (bbox[2] + bbox[0]) / 2
        bbox_center_y = (bbox[3] + bbox[1]) / 2
        # todo- what is 'centered'? what size?
        pic_center_x = img_width/2
        pic_center_y = img_height/2
        return abs(pic_center_x-bbox_center_x) <= 65 and abs(pic_center_y-bbox_center_y) <= 65
