from PIL import Image, ImageDraw
import os
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask, SegmentationCharMask, CharPolygons
import numpy as np
import torch


class AnswersheetDataset(object):
    def __init__(self,imgs_dir,use_charann=False,gts_dir=None,transforms= None):
        self.image_dir = imgs_dir
        self.use_charann = use_charann
        self.img_list = os.listdir(self.image_dir)
        self.img_list.remove('_annotations.csv')
        self.path_list = list(map(lambda x:os.path.join('/content/valid/',x) ,self.img_list))
    
    def __len__(self):
        return len(os.listdir(self.image_dir))-1

    def __getitem__(self,idx):
        img = Image.open(self.path_list[idx]).convert('RGB')
        img = torch.from_numpy(np.array(img))
        img = img.permute(2,0,1)
        img = img.type(torch.FloatTensor)
        target = None
        path = self.path_list[idx]
        return  img,target,path

    
    
