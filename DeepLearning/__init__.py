import os
import sys
from .models import *

# 获取数据集的根目录路径
DATASETS_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')

if sys.platform == 'win32':
    DATASETS_ROOT_PATH = DATASETS_ROOT_PATH.replace(os.sep, '/')
