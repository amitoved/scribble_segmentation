import os

import numpy as np
from matplotlib.pyplot import cm

_constants_path = os.path.realpath(__file__)
PROJECT_DIR = os.path.realpath(os.path.dirname(_constants_path))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

BACKGROUND = 'background'
FOREGROUND = 'foreground'

classes_order = [BACKGROUND, FOREGROUND]
classes = {cls: i for i, cls in enumerate(classes_order)}

n_classes = len(classes.keys())
alpha = 0.5
class_colors = cm.rainbow(np.linspace(0, 1, len(classes_order)))
BRUSH_SIZES = ['2', '5', '10', '15', '30']

SEED = 42
