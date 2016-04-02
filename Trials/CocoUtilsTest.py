from CocoUtils import *
import os


def test_img(img_id, img_path):
    img = coco_utils.coco.loadImgs(img_id)[0]
    anns = coco_utils.get_img_annotations(img_id)
    if not coco_utils.are_legal_anotations(anns):
        print 'illegal annotations for picture %s' % img_id
        return 0

    coco_utils.show_annotations(img_path, anns)

    for ann_num in range(len(anns)):
        centered_text = ''
        # if coco_utils.is_segmentation_centered(anns[ann_num], img['width'], img['height']):
        #     centered_text = 'center'
        im = coco_utils.get_annotation_image(anns[ann_num], img['width'], img['height'])
        im_path = str.format('../Results/%s_%s_%s.jpg' % (img_id, ann_num, centered_text))
        im.save(im_path, 'JPEG')

    return len(anns)


# these lines causes errors with ntdll. might also happen in showAnns, like in image 262148
# as an alternative, I use ImageDraw to manually calculate the mask

# Rs = frPyObjects(ann['segmentation'], img['height'], img['width'])
# masks = decode(Rs)

imgs_to_test = 10
imgs_path = '../../annotations/images/'
coco_utils = CocoUtils('../..', 'train2014')
success = 0
current_img_i = 0
while success < imgs_to_test:
    pic_data = coco_utils.coco.imgs.items()[current_img_i]
    pic_id = pic_data[0]
    pic_path = imgs_path + pic_data[1]['file_name']
    if os.path.isfile(pic_path):
        anns_found = test_img(pic_id, pic_path)
        print('Finished %s annotations in picture %s in %s' % (anns_found, pic_id, pic_path))
        success += 1
    current_img_i += 1
