import numpy as np
import glob
import shutil

source_dir = '../Results/neg-train'
dest_dir = '../Predictions/train'

ex_paths = glob.glob('%s/*-im.png' % source_dir)
np.random.shuffle(ex_paths)
selected = ex_paths[0:8000]

for sel in selected:
    shutil.copy(sel, dest_dir)
    shutil.copy(sel.replace('im', 'mask'), dest_dir)
