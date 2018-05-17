# -*- coding: utf-8 -*-

import Augmentor
p = Augmentor.Pipeline("./Abelmoschus_esculentus", output_directory="./out")

p.zoom(probability=0.1, min_factor=1.1, max_factor=1.3)
p.histogram_equalisation(0.2)
p.flip_left_right(probability=0.1)
p.flip_top_bottom(0.1)
p.rotate(probability=0.2, max_left_rotation=20, max_right_rotation=20)
p.rotate90(0.1)
p.rotate270(0.1)
p.shear(probability=0.2, max_shear_left=10, max_shear_right=10)
p.skew(probability=0.1, magnitude=0.6)
p.skew_tilt(probability=0.2, magnitude=0.6)
p.skew_corner(probability=0.1, magnitude=0.6)
p.random_distortion(probability=0.3, grid_height=4, grid_width=4, magnitude=4)
p.crop_random(0.2, 0.8, randomise_percentage_area=False)
p.scale(0.1, 1.1)
# SIZE = 4164 * 4
SIZE = 200
p.sample(SIZE)
