from math import pow

# loss
score_output_lambda = 1./32
seg_output_lambda = 1

# mask
mask_pic_true_color = 255
mask_pic_false_color = 0
mask_threshold = 0.1

# pic sizes
input_pic_size = 224
output_mask_size = 56

# examples generation
max_centered_object_dimension = 128
translation_shift = 16
scale_deformation = pow(2.0, 0.25)
