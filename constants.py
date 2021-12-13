import os

import numpy as np
from matplotlib.pyplot import cm

_constants_path = os.path.realpath(__file__)
PROJECT_DIR = os.path.realpath(os.path.dirname(_constants_path))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

BACKGROUND = 'background'
SPINE = 'spine'
LIVER = 'liver'
PELVIS = 'pelvis'

classes_order = [BACKGROUND, SPINE, LIVER, PELVIS]

classes = {BACKGROUND: 0,
           SPINE: 1,
           LIVER: 2,
           PELVIS: 3}

n_classes = len(classes.keys())
alpha = 0.5
class_colors = cm.rainbow(np.linspace(0, 1, len(classes_order)))
BRUSH_SIZES = ['1', '5', '10', '20', '50']