import os

_constants_path = os.path.realpath(__file__)
PROJECT_DIR = os.path.realpath(os.path.dirname(_constants_path))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

BACKGROUND = 'background'
FOREGROUND = 'foreground'

classes = {BACKGROUND: 0,
           FOREGROUND: 1}
