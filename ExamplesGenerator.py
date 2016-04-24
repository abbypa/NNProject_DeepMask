from CocoUtils import *
from PIL import Image, ImageOps
import os
from Constants import input_pic_size, max_centered_object_dimension, translation_shift, scale_deformation, \
    negative_ex_min_offset, negative_ex_min_scale


class ExamplesGenerator(object):
    def __init__(self, data_dir, data_type, input_dir, positive_output_dir, negative_output_dir, debug=False):
        self.coco_utils = CocoUtils(data_dir, data_type)
        self.images_dir = '%s/annotations/%s/' % (data_dir, input_dir)
        self.window_size = input_pic_size
        self.max_object_size = max_centered_object_dimension
        self.debug = debug
        self.positive_output_dir = positive_output_dir
        self.negative_output_dir = negative_output_dir
        self.empty_mask = Image.new('L', (input_pic_size, input_pic_size), mask_pic_true_color)

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
                new_can_patch = self.create_positive_examples_from_segmentation(segmentation, pic_id,
                                                                                pic_patch, im_arr, stats)
                if new_can_patch is not None:
                    canonical_seg_patches.append(new_can_patch)

            if not len(canonical_seg_patches) == 0:
                negatives = self.create_negative_examples_from_picture(canonical_seg_patches, pic_id, pic_patch, im_arr,
                                                                       stats)
                stats.negatives_generated += negatives

            if (examples_to_generate is not None
               and examples_to_generate <= stats.positives_generated + stats.negatives_generated):
                print 'Generated enough examples- stopping'
                break
        return stats

    def create_positive_examples_from_segmentation(self, segmentation, pic_id, pic_patch, im_arr, stats):
        seg_id = segmentation['id']

        # bbs - [x y w h]
        bbox = segmentation['bbox']
        bbox_patch = Patch(x_min=bbox[0], width=bbox[2], y_min=bbox[1], height=bbox[3])
        [seg_width, seg_height] = bbox_patch.size()

        if self.segment_size_not_right(seg_width, seg_height, seg_id, pic_id, stats):
            return None

        [seg_center_x, seg_center_y] = bbox_patch.center()

        seg_patch = Patch(x_min=seg_center_x - self.window_size / 2, y_min=seg_center_y - self.window_size / 2,
                          height=self.window_size, width=self.window_size)

        if self.patch_exceeds_pic(seg_patch, pic_patch):
            if self.debug:
                print 'segment %d in picture %d cannot be centered (too close to the edges)' % (seg_id, pic_id)
            stats.seg_too_close_to_edges += 1
            return None

        [pic_width, pic_height] = pic_patch.size()
        seg_im = self.coco_utils.get_annotation_image(segmentation, pic_width, pic_height)

        positives = self.create_positive_canonical_and_noisy_examples_from_mask(im_arr, seg_im, seg_patch, pic_patch,
                                                                                bbox_patch, pic_id, seg_id, stats)
        stats.seg_success += 1
        stats.positives_generated += positives

        return seg_patch

    def create_positive_canonical_and_noisy_examples_from_mask(self, im_arr, full_seg_im, orig_seg_patch, pic_patch,
                                                               bbox_patch, pic_id, seg_id, stats):
        created_examples = 0

        offsets = [-translation_shift, 0, translation_shift]
        scales = [pow(2.0, scale_deformation), 1, pow(2.0, -scale_deformation)]

        [orig_patch_center_x, orig_patch_center_y] = orig_seg_patch.center()
        [orig_patch_width, orig_patch_height] = orig_seg_patch.size()

        for scale_i in range(len(scales)):
            for x_offset_i in range(len(offsets)):
                for y_offset_i in range(len(offsets)):

                    new_patch_width = orig_patch_width * scales[scale_i]
                    new_patch_height = orig_patch_height * scales[scale_i]
                    new_patch_min_x = orig_patch_center_x - new_patch_width / 2 + offsets[x_offset_i]
                    new_patch_min_y = orig_patch_center_y - new_patch_height / 2 + offsets[y_offset_i]
                    new_patch = Patch(new_patch_min_x, new_patch_width, new_patch_min_y, new_patch_height)

                    if self.patch_exceeds_pic(new_patch, pic_patch):
                        stats.pos_noisy_seg_too_close_to_edges += 1
                        continue

                    if self.patch_exceeds_seg(new_patch, bbox_patch):
                        # this will not happen with the default constants (input size, max object dimension)
                        stats.pos_noisy_seg_cuts_seg += 1
                        continue

                    img_path = self.create_path(self.positive_output_dir, 'pos', 'im', pic_id, seg_id, x_offset_i,
                                                y_offset_i, scale_i)
                    patch_im = self.create_and_save_image_patch(im_arr, new_patch, img_path)

                    mask_path = self.create_path(self.positive_output_dir, 'pos', 'mask', pic_id, seg_id,
                                                 x_offset_i, y_offset_i, scale_i)
                    patch_seg_im = self.create_and_save_mask(full_seg_im, new_patch, mask_path)

                    self.create_and_save_mirror(self.positive_output_dir, 'pos', patch_seg_im, patch_im, pic_id,
                                                seg_id, x_offset_i, y_offset_i, scale_i)

                    created_examples += 2  # example and mirror
        return created_examples

    def create_and_save_mirror(self, base_dir, ex_type, mask, im_patch, pic_id, seg_id, x_offset, y_offset, scale):
        mir_im = ImageOps.mirror(im_patch)
        mir_im.save(self.create_path(base_dir, ex_type, 'mir-im', pic_id, seg_id, x_offset, y_offset, scale))
        mir_mask = ImageOps.mirror(mask)
        mir_mask.save(self.create_path(base_dir, ex_type, 'mir-mask', pic_id, seg_id, x_offset, y_offset, scale))

    def create_path(self, base_dir, ex_type, im_type, pic_id, seg_id, offset_x, offset_y, scale):
        return str('%s/%s-%d-%d-%d-%d-%d-%s.png' % (base_dir, ex_type, pic_id, seg_id, offset_x, offset_y,
                                                    scale, im_type))

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
        patch_im = patch_im.resize((self.window_size, self.window_size), Image.ANTIALIAS)
        patch_im.save(img_path)
        return patch_im

    def create_and_save_mask(self, full_seg_im, new_patch, mask_path):
        new_patch_x_max = new_patch.x_min + new_patch.width  # not inclusive
        new_patch_y_max = new_patch.y_min + new_patch.height
        patch_seg_im = full_seg_im.crop((new_patch.x_min, new_patch.y_min, new_patch_x_max, new_patch_y_max))
        patch_seg_im = patch_seg_im.resize((self.window_size, self.window_size), Image.ANTIALIAS)
        patch_seg_im.save(mask_path)
        return patch_seg_im

    def create_negative_examples_from_picture(self, canonical_seg_patches, pic_id, pic_patch, im_arr, stats):
        curr_ex_id = 0
        examples_generated = 0
        can_seg_patch_centers = [seg.center() for seg in canonical_seg_patches]

        # including neutral translation to allow one dimension noise
        offsets = [-negative_ex_min_offset * 1.5, -negative_ex_min_offset, 0,
                   negative_ex_min_offset, negative_ex_min_offset * 1.5]
        scales = [pow(2, -negative_ex_min_scale-0.5), pow(2, -negative_ex_min_scale),
                  pow(2, negative_ex_min_scale), pow(2, negative_ex_min_scale+0.5)]

        for can_seg_patch in canonical_seg_patches:
            [can_patch_center_x, can_patch_center_y] = can_seg_patch.center()

            for xoi, x_offset in enumerate(offsets):
                for yoi, y_offset in enumerate(offsets):

                    if x_offset == 0 and y_offset == 0:
                        continue  # no alteration

                    neg_patch = Patch(x_min=can_seg_patch.x_min + x_offset,
                                      y_min=can_seg_patch.y_min + y_offset,
                                      width=can_seg_patch.width, height=can_seg_patch.height)

                    if self.is_close_to_other_patches(neg_patch.center(), can_seg_patch_centers):
                        stats.neg_seg_too_close_to_other_segs += 1
                        continue

                    if self.patch_exceeds_pic(neg_patch, pic_patch):
                        stats.neg_seg_too_close_to_edges += 1
                        continue
                    neg_path = self.create_path(self.negative_output_dir, 'neg', 'im', pic_id, curr_ex_id,
                                                xoi, yoi, 0)
                    im_patch = self.create_and_save_image_patch(im_arr, neg_patch, neg_path)
                    mask_path = self.create_path(self.negative_output_dir, 'neg', 'mask', pic_id, curr_ex_id,
                                                 xoi, yoi, 0)
                    self.empty_mask.save(mask_path)
                    self.create_and_save_mirror(self.negative_output_dir, 'neg', self.empty_mask, im_patch,
                                                pic_id, curr_ex_id, xoi, yoi, 0)
                    curr_ex_id += 1
                    examples_generated += 2  # pic + mirror

            # scale with aspect ratio kept- otherwise we might scale the lesser dimension, and the mask is legal
            for scale_i, scale in enumerate(scales):

                neg_patch_width = can_seg_patch.width * scale
                neg_patch_height = can_seg_patch.height * scale
                neg_patch_min_x = can_patch_center_x - neg_patch_width / 2
                neg_patch_min_y = can_patch_center_y - neg_patch_height / 2
                neg_patch = Patch(x_min=neg_patch_min_x, width=neg_patch_width,
                                  y_min=neg_patch_min_y, height=neg_patch_height)

                # if we only scale, we do allow the center to be close to another seg
                if self.patch_exceeds_pic(neg_patch, pic_patch):
                    stats.neg_seg_too_close_to_edges += 1
                    continue
                neg_path = self.create_path(self.negative_output_dir, 'neg', 'im', pic_id, curr_ex_id, 0, 0, scale_i)
                im_patch = self.create_and_save_image_patch(im_arr, neg_patch, neg_path)
                mask_path = self.create_path(self.negative_output_dir, 'neg', 'mask', pic_id, curr_ex_id, 0, 0, scale_i)
                self.empty_mask.save(mask_path)
                self.create_and_save_mirror(self.negative_output_dir, 'neg', self.empty_mask, im_patch,
                                            pic_id, curr_ex_id, 0, 0, scale_i)
                curr_ex_id += 1
                examples_generated += 2  # pic + mirror

        return examples_generated

    def is_close_to_other_patches(self, seg_center, other_seg_centers):
        # check seg_center x and y distance from the other centers.
        # each pair must have a minimal distance in at least on dimension
        for other_seg_center in other_seg_centers:
            if(abs(seg_center[0] - other_seg_center[0]) < negative_ex_min_offset
               and abs(seg_center[1] - other_seg_center[1]) < negative_ex_min_offset):
                return True
        return False


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

        self.pos_noisy_seg_too_close_to_edges = 0
        self.pos_noisy_seg_cuts_seg = 0

        self.neg_seg_too_close_to_edges = 0
        self.neg_seg_too_close_to_other_segs = 0

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
                   '\t\tpos\n'
                   '\t\t\tpos noisy seg too close to edges: %d\n'
                   '\t\t\tpos noisy seg cuts seg: %d\n'
                   '\t\tneg\n'
                   '\t\t\tneg seg too close to edges: %d\n'
                   '\t\t\tneg seg too close to other segs: %d\n'
                   '\tpositive examples generated: %d\n'
                   '\tnegative examples generated: %d\n'
                   % (self.img_not_found, self.img_exists, self.img_with_illegal_annotations,
                      self.img_with_legal_annotations, self.seg_too_big, self.seg_too_small,
                      self.seg_too_close_to_edges, self.seg_success,
                      self.pos_noisy_seg_too_close_to_edges, self.pos_noisy_seg_cuts_seg,
                      self.neg_seg_too_close_to_edges, self.neg_seg_too_close_to_other_segs,
                      self.positives_generated, self.negatives_generated))


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


# eg = ExamplesGenerator('..', 'val2014', 'images_val', 'Results/pos-val', 'Results/neg-val')
eg = ExamplesGenerator('..', 'train2014', 'images_train', 'Results/pos-train', 'Results/neg-train')
stats_res = eg.generate_examples()
print stats_res
