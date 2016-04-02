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

    def get_images_data(self):
        # each item is image_id, image_file_name
        return [pic_data[1] for pic_data in self.coco.imgs.items()]