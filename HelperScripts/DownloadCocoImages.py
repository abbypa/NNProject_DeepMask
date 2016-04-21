from CocoUtils import *
import os

data_dir = '../..'
coco_utils = CocoUtils(data_dir, 'train2014')
image_ids_and_names = coco_utils.get_images_data()

images_dir = '%s/annotations/images/' % data_dir

for pic_data in image_ids_and_names:
    pic_id = pic_data['id']
    pic_path = images_dir + pic_data['file_name']
    if not os.path.isfile(pic_path):
        try:
            coco_utils.coco.download(images_dir, [pic_id])
        except:
            pass  # skip
