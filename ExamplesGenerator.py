from CocoUtils import *
from PIL import Image, ImageOps
import os
from Constants import input_pic_size, max_centered_object_dimension, translation_shift, scale_deformation, \
    negative_ex_min_offset, negative_ex_min_scale


class ExamplesGenerator(object):
    def __init__(self, data_dir, data_type, positive_output_dir, negative_output_dir, debug=False):
        self.coco_utils = CocoUtils(data_dir, data_type)
        self.images_dir = '%s/annotations/images/' % data_dir
        self.window_size = input_pic_size
        self.max_object_size = max_centered_object_dimension
        self.debug = debug
        self.positive_output_dir = positive_output_dir
        self.negative_output_dir = negative_output_dir

    def generate_examples(self, examples_to_generate=None):
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

            pic_patch = Patch(x_min=0, y_min=0, height=pic_data['height'], width=pic_data['width'])
            im_arr = io.imread(pic_path)

            canonical_seg_patches = []
            for segmentation in annotations:
                canonical_seg_patches.append(self.create_positive_examples_from_segmentation(segmentation, pic_id,
                                                                                             pic_patch, im_arr, stats))

            negatives = self.create_negative_examples_from_picture(canonical_seg_patches, pic_id, pic_patch, im_arr)
            stats.negatives_generated += negatives

            if examples_to_generate <= stats.positives_generated + stats.negatives_generated:
                # generated enough
                break
        return stats

    def create_positive_examples_from_segmentation(self, segmentation, pic_id, pic_patch, im_arr, stats):
        seg_id = segmentation['id']

        # bbs - [x y w h]
        bbox = segmentation['bbox']
        bbox_patch = Patch(x_min=bbox[0], width=bbox[2], y_min=bbox[1], height=bbox[3])
        [seg_width, seg_height] = bbox_patch.size()

        if self.segment_size_not_right(seg_width, seg_height, seg_id, pic_id, stats):
            return

        [seg_center_x, seg_center_y] = bbox_patch.center()

        seg_patch = Patch(x_min=seg_center_x - self.window_size / 2, y_min=seg_center_y - self.window_size / 2,
                          height=self.window_size, width=self.window_size)

        if self.patch_exceeds_pic(seg_patch, pic_patch):
            if self.debug:
                print 'segment %d in picture %d cannot be centered (too close to the edges)' % (seg_id, pic_id)
            stats.seg_too_close_to_edges += 1
            return

        [pic_width, pic_height] = pic_patch.size()
        seg_im = self.coco_utils.get_annotation_image(segmentation, pic_width, pic_height)

        positives = self.create_positive_canonical_and_noisy_examples_from_mask(im_arr, seg_im, seg_patch, pic_patch, bbox_patch,
                                                                    pic_id, seg_id)
        stats.seg_success += 1
        stats.positives_generated += positives

        return seg_patch

    def create_positive_canonical_and_noisy_examples_from_mask(self, im_arr, full_seg_im, orig_seg_patch, pic_patch,
                                                               bbox_patch, pic_id, seg_id):
        created_examples = 0

        offsets = [-translation_shift, 0, translation_shift]
        scales = [pow(2.0, scale_deformation), 1, pow(2.0, -scale_deformation)]

        [orig_patch_center_x, orig_patch_center_y] = orig_seg_patch.center()
        [orig_patch_width, orig_patch_height] = orig_seg_patch.size()

        for x_scale_i in range(len(scales)):
            for y_scale_i in range(len(scales)):
                for x_offset_i in range(len(offsets)):
                    for y_offset_i in range(len(offsets)):

                        new_patch_width = orig_patch_width * scales[x_scale_i]
                        new_patch_height = orig_patch_height * scales[y_scale_i]
                        new_patch_min_x = orig_patch_center_x - new_patch_width / 2 + offsets[x_offset_i]
                        new_patch_min_y = orig_patch_center_y - new_patch_height / 2 + offsets[y_offset_i]
                        new_patch = Patch(new_patch_min_x, new_patch_width, new_patch_min_y, new_patch_height)

                        if self.patch_exceeds_pic(new_patch, pic_patch):
                            print 'exceeding pic size'
                            continue

                        if self.patch_exceeds_seg(new_patch, bbox_patch):
                            # this will not happen with the default constants (input size, max object dimension)
                            print 'exceeding: xs %s ys %s xo %s yo %s' % (x_scale_i, y_scale_i, x_offset_i, y_offset_i)
                            continue

                        img_path = self.create_path('im', pic_id, seg_id,x_offset_i, y_offset_i, x_scale_i, y_scale_i)
                        patch_im = self.create_and_save_image_patch(im_arr, new_patch, img_path)

                        mask_path = self.create_path('mask', pic_id, seg_id, x_offset_i, y_offset_i, x_scale_i, y_scale_i)
                        patch_seg_im = self.create_and_save_mask(full_seg_im, new_patch, mask_path)

                        self.create_and_save_mirror(patch_seg_im, patch_im, pic_id, seg_id, x_offset_i, y_offset_i,
                                                    x_scale_i, y_scale_i)

                        created_examples += 2  # example and mirror
        return created_examples

    def create_and_save_mirror(self, mask, im_patch, pic_id, seg_id, x_offset, y_offset, x_scale, y_scale):
        mir_im = ImageOps.mirror(im_patch)
        mir_im.save(self.create_path('mir-im', pic_id, seg_id, x_offset, y_offset, x_scale, y_scale))
        mir_mask = ImageOps.mirror(mask)
        mir_mask.save(self.create_path('mir-mask', pic_id, seg_id, x_offset, y_offset, x_scale, y_scale))

    def create_path(self, im_type, pic_id, seg_id, offset_x, offset_y, x_scale, y_scale):
        return str('%s/%d-%d-%d-%d-%d-%d-%s.png' % (self.positive_output_dir, pic_id, seg_id, offset_x, offset_y,
                                                    x_scale, y_scale, im_type))

    def patch_exceeds_pic(self, seg_patch, pic_patch):
        return not pic_patch.contains(seg_patch)

    def patch_exceeds_seg(self, seg_patch, bbox_patch):
        return not seg_patch.contains(bbox_patch)

    def segment_size_not_right(self, seg_width, seg_height, seg_id, pic_id, stats):
        max_dim = max(seg_height, seg_width)
        if max_dim > self.max_object_size:
            if self.debug:
                print 'segment %d in picture %d is too big' % (seg_id, pic_id)
            stats.seg_too_big += 1
            return True

        if max_dim < self.max_object_size:
            if self.debug:
                print 'segment %d in picture %d is too small' % (seg_id, pic_id)
            stats.seg_too_small += 1
            return True
        return False

    def create_and_save_image_patch(self, im_arr, new_patch, img_path):
        new_patch_x_max = new_patch.x_min + new_patch.width  # not inclusive
        new_patch_y_max = new_patch.y_min + new_patch.height
        patch_im_arr = im_arr[new_patch.y_min:new_patch_y_max, new_patch.x_min:new_patch_x_max]
        patch_im = Image.fromarray(patch_im_arr)
        patch_im = patch_im.resize((self.window_size, self.window_size))
        patch_im.save(img_path)
        return patch_im

    def create_and_save_mask(self, full_seg_im, new_patch, mask_path):
        new_patch_x_max = new_patch.x_min + new_patch.width  # not inclusive
        new_patch_y_max = new_patch.y_min + new_patch.height
        patch_seg_im = full_seg_im.crop((new_patch.x_min, new_patch.y_min, new_patch_x_max, new_patch_y_max))
        patch_seg_im = patch_seg_im.resize((self.window_size, self.window_size))
        patch_seg_im.save(mask_path)
        return patch_seg_im

    def create_negative_examples_from_picture(self, canonical_seg_patches, pic_id, pic_patch, im_arr):
        curr_ex_id = 0
        offsets = [-negative_ex_min_offset, negative_ex_min_offset]
        scales = [pow(2, -negative_ex_min_scale), pow(2, negative_ex_min_scale)]
        for can_seg_patch in canonical_seg_patches:
            for x_offset_i in offsets:
                for y_offset_i in offsets:
                    neg_patch = Patch(x_min=can_seg_patch.x_min + offsets[x_offset_i],
                                      y_min=can_seg_patch.y_min + offsets[y_offset_i],
                                      width=can_seg_patch.width, height=can_seg_patch.height)
                    # TODO- check not exceeding picture
                    # TODO- check closenes to other segs
                    neg_path = self.create_path('im', pic_id, curr_ex_id, x_offset_i, y_offset_i, 0, 0)
                    self.create_and_save_image_patch(im_arr, neg_patch, neg_path)
                    # TODO- create all-ones mask
                    curr_ex_id += 1
        # TODO same for scales (mix?)
        return 0

class ExampleGeneratorStats(object):
    def __init__(self):
        self.img_not_found = 0
        self.img_exists = 0
        self.img_with_illegal_annotations = 0
        self.img_with_legal_annotations = 0

        self.seg_too_big = 0
        self.seg_too_small = 0
        self.seg_too_close_to_edges = 0
        self.seg_success = 0

        self.positives_generated = 0
        self.negatives_generated = 0

    def __str__(self):
        return str('imgs not found: %d\n'
                   'imgs found: %d\n'
                   '\timgs with illegal annotations: %d\n'
                   '\timgs with legal annotations: %d\n'
                   '\t\tseg too big: %d\n'
                   '\t\tseg too small: %d\n'
                   '\t\tseg too close to edges: %d\n'
                   '\t\tseg success: %d\n'
                   '\t\tpositive examples generated: %d\n'
                   '\t\tnegative examples generated: %d\n'
                   % (self.img_not_found, self.img_exists, self.img_with_illegal_annotations,
                      self.img_with_legal_annotations, self.seg_too_big, self.seg_too_small,
                      self.seg_too_close_to_edges, self.seg_success, self.positives_generated, self.negatives_generated))


class Patch(object):
    def __init__(self, x_min, width, y_min, height):
        self.x_min = int(round(x_min))
        self.width = int(round(width))
        self.y_min = int(round(y_min))
        self.height = int(round(height))

    def contains(self, contained_patch):
        return (self.x_min <= contained_patch.x_min and self.y_min <= contained_patch.y_min
                and self.x_min + self.width >= contained_patch.x_min + contained_patch.width
                and self.y_min + self.height >= contained_patch.y_min + contained_patch.height)

    def center(self):
        return [self.x_min + self.width / 2, self.y_min + self.height / 2]

    def size(self):
        return [self.width, self.height]


eg = ExamplesGenerator('..', 'train2014', 'Results/pos', 'Results/neg')
stats_res = eg.generate_examples()
print stats_res
