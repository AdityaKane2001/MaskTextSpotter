from PIL import Image, ImageDraw
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, SegmentationCharMask, CharPolygons
import numpy as np
import torch


class AnswersheetDataset(object):
    def __init__(self,imgs_dir,use_charann=False,gts_dir=None):
        self.image_dir = imgs_dir
        self.use_charann = use_charann
        
