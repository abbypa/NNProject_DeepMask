from CocoUtils import *
from PIL import Image
import os


class ExamplesGenerator(object):
    def __init__(self, data_dir, data_type, output_dir, debug=False):
        self.coco_utils = CocoUtils(data_dir, data_type)
        self.images_dir = '%s/annotations/images/' % data_dir
        self.window_size = 224
        self.max_object_size = 128
        self.debug = debug
        self.output_dir = output_dir

    def generate_positive_examples(self):
        stats = ExampleGeneratorStats()

        image_ids_and_names = self.coco_utils.get_images_data()

        for pic_data in image_ids_and_names:
            pic_id = pic_data['id']
            pic_path = self.images_dir + pic_data['file_name']

            if not os.path.isfile(pic_path):
                stats.img_not_found += 1
                if self.debug:
                    print 'image %d does not exist' % pic_id
                continue  # img does not exist

            stats.img_exists += 1

            annotations = self.coco_utils.get_img_annotations(pic_id)
            if not self.coco_utils.are_legal_anotations(annotations):
                if self.debug:
                    print 'illegal annotations for picture %s' % pic_id
                stats.img_with_illegal_annotations += 1
                continue

            stats.img_with_legal_annotations += 1

            for segmentation in annotations:
                self.create_positive_example(pic_data, segmentation, pic_path, pic_id, stats)

        return stats

    def create_positive_example(self, pic_data, segmentation, pic_path, pic_id, stats):

        seg_id = segmentation['id']

        # bbs - [x y w h]
        bbox = segmentation['bbox']
        seg_height = bbox[3]
        seg_width = bbox[2]

        if max(seg_height, seg_width) > self.max_object_size:
            if self.debug:
                print 'segment %d in picture %d is too big' % (seg_id, pic_id)
            stats.seg_too_big += 1
            return

        pic_height = pic_data['height']
        pic_width = pic_data['width']

        seg_center_x = bbox[0] + seg_width / 2
        seg_center_y = bbox[1] + seg_height / 2

        patch_min_x = seg_center_x - self.window_size/2
        patch_max_x = seg_center_x + self.window_size/2
        patch_min_y = seg_center_y - self.window_size/2
        patch_max_y = seg_center_y + self.window_size/2

        if patch_min_x < 0 or patch_min_y < 0 or patch_max_x > pic_width or patch_max_y > pic_height:
            if self.debug:
                print 'segment %d in picture %d cannot be centered (too close to the edges)' % (seg_id, pic_id)
            stats.seg_too_close_to_edges += 1
            return

        im_arr = io.imread(pic_path)
        patch_im_arr = im_arr[patch_min_y:patch_max_y, patch_min_x:patch_max_x]
        patch_im = Image.fromarray(patch_im_arr)
        output_path = str('%s/%d-%d-im.png' % (self.output_dir, pic_id, seg_id))
        patch_im.save(output_path)
        if self.debug:
            patch_im.show()

        # get_annotation_mask
        seg_im = self.coco_utils.get_annotation_image(segmentation, pic_width, pic_height)
        patch_seg_im = seg_im.crop((int(patch_min_x), int(patch_min_y), int(patch_max_x), int(patch_max_y)))
        output_path = str('%s/%d-%d-mask.png' % (self.output_dir, pic_id, seg_id))
        patch_seg_im.save(output_path)
        if self.debug:
            patch_seg_im.show()

        stats.seg_success += 1


class ExampleGeneratorStats(object):
    def __init__(self):
        self.img_not_found = 0
        self.img_exists = 0
        self.img_with_illegal_annotations = 0
        self.img_with_legal_annotations = 0

        self.seg_too_big = 0
        self.seg_too_close_to_edges = 0
        self.seg_success = 0

    def __str__(self):
        return str('imgs not found: %d\n'
                   'imgs found: %d\n'
                   '\timgs with illegal annotations: %d\n'
                   '\timgs with legal annotations: %d\n'
                   '\t\tseg too big: %d\n'
                   '\t\tseg too close to edges: %d\n'
                   '\t\tseg success: %d\n'
                   % (self.img_not_found, self.img_exists, self.img_with_illegal_annotations,
                      self.img_with_legal_annotations, self.seg_too_big, self.seg_too_close_to_edges, self.seg_success))


eg = ExamplesGenerator('..', 'train2014', 'Results')
stats_res = eg.generate_positive_examples()
print stats_res
